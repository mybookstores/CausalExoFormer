from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_CausalExoFormer(Exp_Long_Term_Forecast):
    """
    Specialized experiment class for CausalExoFormer.
    Extends Exp_Long_Term_Forecast with:
    - Augmented Lagrangian updates for DAG constraint
    - Temperature annealing for causal adjacency
    - Causal auxiliary losses in training
    - Causal graph output and visualization
    - V2: Causal warmup, ramp-up, independent learning rate for causal params
    """

    def __init__(self, args):
        super(Exp_CausalExoFormer, self).__init__(args)
        # Augmented Lagrangian variables
        self.alpha_lagrangian = 0.0  # Lagrange multiplier
        self.rho = args.lambda_dag  # Penalty coefficient
        # Temperature annealing
        self.temp_decay_rate = (args.temp_min / args.temp_init) ** (1.0 / max(args.train_epochs, 1))
        # V2: Warmup and ramp-up settings
        self.causal_warmup_epochs = getattr(args, 'causal_warmup_epochs', 5)
        self.causal_rampup_epochs = getattr(args, 'causal_rampup_epochs', 5)
        self.causal_lr_scale = getattr(args, 'causal_lr_scale', 10.0)

    def _get_raw_model(self):
        """Get the underlying model, unwrapping DataParallel if needed."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # V2: Build optimizer with separate param groups for causal and prediction params
        model_optim = self._build_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            causal_loss_epoch = []

            self.model.train()
            epoch_time = time.time()

            # V2: Compute causal loss weight based on warmup/ramp-up schedule
            causal_weight = self._get_causal_weight(epoch)

            # V2: Update causal param learning rate based on warmup
            self._update_causal_lr(model_optim, epoch)

            # Track last h_W for Lagrangian update
            last_h_W = 0.0

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        pred_loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    pred_loss = criterion(outputs, batch_y)

                # Compute causal auxiliary losses
                # NOTE: For large multivariate datasets (e.g., traffic), DAG constraint computation
                # can be expensive (torch.matrix_exp). When causal_weight==0, these losses do not
                # contribute to optimization, so skip to avoid unnecessary overhead.
                if causal_weight > 0:
                    causal_losses = self._get_raw_model().get_causal_losses()
                    sparse_loss = causal_losses['sparse_loss']
                    h_W = causal_losses['dag_constraint']
                    consist_loss = causal_losses['consist_loss']
                    exo_graph = causal_losses.get('exo_graph')
                else:
                    zero = torch.zeros((), device=self.device)
                    sparse_loss = zero
                    h_W = zero
                    consist_loss = zero
                    exo_graph = None

                # V2: Total loss with causal_weight modulation
                loss = pred_loss
                if causal_weight > 0:
                    loss = (loss
                            + causal_weight * self.args.lambda_sparse * sparse_loss
                            + causal_weight * (self.rho / 2.0) * h_W * h_W
                            + causal_weight * self.alpha_lagrangian * h_W
                            + causal_weight * self.args.lambda_consist * consist_loss)

                train_loss.append(pred_loss.item())
                causal_loss_epoch.append({
                    'sparse': sparse_loss.item(),
                    'h_W': h_W.item(),
                    'consist': consist_loss.item()
                })
                last_h_W = h_W.item()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | pred_loss: {2:.7f} | h_W: {3:.6f} | sparse: {4:.4f} | consist: {5:.6f} | causal_w: {6:.4f}".format(
                        i + 1, epoch + 1, pred_loss.item(), h_W.item(), sparse_loss.item(), consist_loss.item(), causal_weight))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # V2: Augmented Lagrangian update only after warmup
            if epoch >= self.causal_warmup_epochs:
                self.alpha_lagrangian = self.alpha_lagrangian + self.rho * last_h_W
                # V2: More conservative growth factor (2.0 instead of 10.0)
                self.rho = min(self.rho * min(self.args.dag_penalty_growth, 2.0), self.args.dag_penalty_max)

            # Temperature annealing
            raw_model = self._get_raw_model()
            current_temp = raw_model.causal_adj.temperature
            new_temp = max(current_temp * self.temp_decay_rate, self.args.temp_min)
            raw_model.causal_adj.temperature = new_temp

            # Print epoch summary with causal info
            avg_h_W = np.mean([c['h_W'] for c in causal_loss_epoch])
            avg_sparse = np.mean([c['sparse'] for c in causal_loss_epoch])
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("  [Causal] h_W: {0:.6f} | Sparse: {1:.4f} | Temp: {2:.4f} | rho: {3:.2f} | alpha: {4:.4f} | weight: {5:.4f}".format(
                avg_h_W, avg_sparse, new_temp, self.rho, self.alpha_lagrangian, causal_weight))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # Print causal structure summary after loading best model
        self._print_causal_summary()

        return self.model

    def _build_optimizer(self):
        """V2: Build optimizer with separate param groups for prediction and causal params."""
        raw_model = self._get_raw_model()
        causal_param_ids = set()
        causal_params = []
        # Collect A_raw parameters from CausalAdjacencyModule
        for name, param in raw_model.named_parameters():
            if (('causal_adj.A_raw' in name) or ('exo_exo_mixer.B_raw' in name)) and param.requires_grad:
                causal_params.append(param)
                causal_param_ids.add(id(param))

        # All other trainable parameters
        pred_params = [p for p in raw_model.parameters() if p.requires_grad and id(p) not in causal_param_ids]

        param_groups = [
            {'params': pred_params, 'lr': self.args.learning_rate},
        ]
        if causal_params:
            # V2: Start with lr=0 during warmup; will be updated per epoch
            initial_causal_lr = 0.0 if self.causal_warmup_epochs > 0 else self.args.learning_rate * self.causal_lr_scale
            param_groups.append({
                'params': causal_params,
                'lr': initial_causal_lr,
                'label': 'causal'
            })

        optimizer = optim.Adam(param_groups)
        return optimizer

    def _get_causal_weight(self, epoch):
        """V2: Compute causal loss weight based on warmup and ramp-up schedule.

        - epoch < causal_warmup_epochs: weight = 0
        - causal_warmup_epochs <= epoch < causal_warmup_epochs + causal_rampup_epochs: linear ramp-up
        - epoch >= causal_warmup_epochs + causal_rampup_epochs: weight = 1.0
        """
        if self.causal_warmup_epochs == 0 and self.causal_rampup_epochs == 0:
            return 1.0  # Backward compatible
        if epoch < self.causal_warmup_epochs:
            return 0.0
        rampup_epoch = epoch - self.causal_warmup_epochs
        if self.causal_rampup_epochs > 0 and rampup_epoch < self.causal_rampup_epochs:
            return float(rampup_epoch + 1) / float(self.causal_rampup_epochs)
        return 1.0

    def _update_causal_lr(self, optimizer, epoch):
        """V2: Update causal param group learning rate based on warmup schedule."""
        for group in optimizer.param_groups:
            if group.get('label') == 'causal':
                if epoch < self.causal_warmup_epochs:
                    group['lr'] = 0.0  # Frozen during warmup
                else:
                    group['lr'] = self.args.learning_rate * self.causal_lr_scale

    def test(self, setting, test=0):
        # Run standard test from parent class
        super().test(setting, test)

        # Save causal graph outputs
        self._save_causal_outputs(setting)

    def _print_causal_summary(self):
        """Print a summary of the learned causal structure."""
        self.model.eval()
        with torch.no_grad():
            raw_model = self._get_raw_model()
            causal_gate = raw_model.causal_adj.get_causal_gate().detach().cpu().numpy()
            lag_dist = raw_model.causal_adj.get_lag_distribution().detach().cpu().numpy()
            C, K = causal_gate.shape

        print("\n" + "=" * 60)
        print("Learned Causal Structure Summary")
        print("=" * 60)
        for c in range(C):
            max_gate = causal_gate[c].max()
            best_lag = np.argmax(lag_dist[c]) * raw_model.lag_step
            significant = "*** SIGNIFICANT ***" if max_gate > 0.5 else ""
            print("  Exo Var {0}: gate_max={1:.4f}, best_lag={2} {3}".format(
                c, max_gate, best_lag, significant))
        print("  [Gate sparsity] mean={0:.4f}, <0.1 ratio={1:.2%}".format(
            causal_gate.mean(), (causal_gate < 0.1).mean()))
        # Exo-exo graph summary
        causal_losses = raw_model.get_causal_losses()
        if causal_losses.get('exo_graph') is not None:
            eg = causal_losses['exo_graph'].numpy()
            print("  [Exo-Exo Graph] mean={0:.4f}, <0.1 ratio={1:.2%}".format(
                eg.mean(), (eg < 0.1).mean()))
        print("=" * 60 + "\n")

    def _save_causal_outputs(self, setting):
        """Save causal graph, lag distribution, exo-exo graph, and optionally generate heatmaps."""
        self.model.eval()
        with torch.no_grad():
            raw_model = self._get_raw_model()
            causal_graph = raw_model.causal_adj.get_causal_gate().detach().cpu().numpy()
            lag_dist = raw_model.causal_adj.get_lag_distribution().detach().cpu().numpy()
            causal_losses = raw_model.get_causal_losses()
            exo_graph = causal_losses.get('exo_graph')

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'causal_graph.npy', causal_graph)
        np.save(folder_path + 'lag_distribution.npy', lag_dist)
        print("Causal graph saved to: {}causal_graph.npy".format(folder_path))
        print("Lag distribution saved to: {}lag_distribution.npy".format(folder_path))
        if exo_graph is not None:
            np.save(folder_path + 'exo_graph.npy', exo_graph.numpy())
            print("Exo-exo graph saved to: {}exo_graph.npy".format(folder_path))

        self._print_causal_summary()

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            n_figs = 3 if exo_graph is not None else 2
            fig, axes = plt.subplots(1, n_figs, figsize=(6 * n_figs, 5))
            if n_figs == 1:
                axes = [axes]
            # Gate heatmap
            im0 = axes[0].imshow(causal_graph, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[0].set_title('Causal Gate G [C×K]')
            axes[0].set_xlabel('Lag Index (×step)')
            axes[0].set_ylabel('Exogenous Variable')
            plt.colorbar(im0, ax=axes[0])
            # Lag distribution heatmap
            im1 = axes[1].imshow(lag_dist, aspect='auto', cmap='Blues')
            axes[1].set_title('Lag Distribution (softmax)')
            axes[1].set_xlabel('Lag Index')
            axes[1].set_ylabel('Exogenous Variable')
            plt.colorbar(im1, ax=axes[1])
            # Exo-exo graph
            if n_figs == 3 and exo_graph is not None:
                eg = exo_graph.numpy()
                im2 = axes[2].imshow(eg, aspect='auto', cmap='Greens', vmin=0, vmax=1)
                axes[2].set_title('Exo-Exo Adjacency B [C×C]')
                axes[2].set_xlabel('Exo Variable')
                axes[2].set_ylabel('Exo Variable')
                plt.colorbar(im2, ax=axes[2])
            plt.tight_layout()
            plt.savefig(folder_path + 'causal_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Causal heatmap saved to: {}causal_heatmap.png".format(folder_path))
        except ImportError:
            print("matplotlib not available, skipping heatmap visualization.")

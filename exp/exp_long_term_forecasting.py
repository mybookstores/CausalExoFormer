from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.losses import tari_loss

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 支持 args.loss 选择: 'MSE' | 'MAE' | 'Huber' | 'SmoothL1' | 'MSE_MAE' | 'Tari'
        loss_name = getattr(self.args, 'loss', 'MSE')
        if loss_name == 'Huber':
            delta = float(getattr(self.args, 'huber_delta', 1.0))
            criterion = nn.HuberLoss(delta=delta)
        elif loss_name == 'SmoothL1':
            beta = float(getattr(self.args, 'smooth_l1_beta', 1.0))
            criterion = nn.SmoothL1Loss(beta=beta)
        elif loss_name == 'MAE':
            criterion = nn.L1Loss()
        elif loss_name == 'MSE_MAE':
            w = float(getattr(self.args, 'mse_mae_weight', 0.5))
            return lambda pred, true: w * nn.functional.mse_loss(pred, true) + (1 - w) * nn.functional.l1_loss(pred, true)
        elif loss_name == 'Tari':
            alpha = float(getattr(self.args, 'tari_alpha', 0.7))
            return lambda pred, true: tari_loss(pred, true, alpha)
        else:
            criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion, return_preds=False):
        total_loss = []
        preds = []
        trues = []
        last_encs = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs_for_loss = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_for_loss = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs_for_loss.detach()
                true = batch_y_for_loss.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())

                if return_preds:
                    outputs_np = outputs[:, -self.args.pred_len:, :].detach().cpu().numpy()
                    batch_y_np = batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy()
                    if vali_data.scale and self.args.inverse:
                        shape = batch_y_np.shape
                        if outputs_np.shape[-1] != batch_y_np.shape[-1]:
                            outputs_np = np.tile(outputs_np, [1, 1, int(batch_y_np.shape[-1] / outputs_np.shape[-1])])
                        outputs_np = vali_data.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y_np = vali_data.inverse_transform(batch_y_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    outputs_np = outputs_np[:, :, f_dim:]
                    batch_y_np = batch_y_np[:, :, f_dim:]

                    batch_x_np = batch_x.detach().cpu().numpy()
                    if vali_data.scale and self.args.inverse:
                        shape_bx = batch_x_np.shape
                        batch_x_np = vali_data.inverse_transform(batch_x_np.reshape(shape_bx[0] * shape_bx[1], -1)).reshape(shape_bx)
                    last_encs.append(batch_x_np[:, -1, f_dim])
                    preds.append(outputs_np)
                    trues.append(batch_y_np)
        total_loss = np.average(total_loss)
        self.model.train()
        if return_preds:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            last_encs = np.concatenate(last_encs, axis=0)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            return total_loss, preds, trues, last_encs
        return total_loss

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

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        onecycle_scheduler = None
        if getattr(self.args, 'use_onecycle', 0):
            onecycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                model_optim,
                max_lr=self.args.learning_rate,
                total_steps=self.args.train_epochs * train_steps,
                pct_start=float(getattr(self.args, 'onecycle_pct_start', 0.3)),
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4,
            )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if getattr(self.args, 'grad_clip', 0.0) > 0:
                        scaler.unscale_(model_optim)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.args.grad_clip))
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if getattr(self.args, 'grad_clip', 0.0) > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.args.grad_clip))
                    model_optim.step()
                if onecycle_scheduler is not None:
                    onecycle_scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path, train_loss=train_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if onecycle_scheduler is None:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        last_encs = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_orig = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y_orig = batch_y_orig.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y_orig.shape
                    if outputs.shape[-1] != batch_y_orig.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y_orig.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_orig = test_data.inverse_transform(batch_y_orig.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y_orig = batch_y_orig[:, :, f_dim:]

                # collect last encoder value for endpoint correction
                batch_x_np = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape_bx = batch_x_np.shape
                    batch_x_np = test_data.inverse_transform(batch_x_np.reshape(shape_bx[0] * shape_bx[1], -1)).reshape(shape_bx)
                last_enc = batch_x_np[:, -1, f_dim]  # [B]
                last_encs.append(last_enc)

                pred = outputs
                true = batch_y_orig

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        last_encs = np.concatenate(last_encs, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        preds = self._apply_fixed_postprocess(preds, last_encs)

        calibration_method = getattr(self.args, 'calibration_method', 'none')
        if calibration_method != 'none':
            vali_data, vali_loader = self._get_data(flag='val')
            criterion = self._select_criterion()
            _, val_preds, val_trues, val_last_encs = self.vali(vali_data, vali_loader, criterion, return_preds=True)
            val_preds = self._apply_fixed_postprocess(val_preds, val_last_encs)
            calibration = self._fit_calibration(val_preds, val_trues, val_last_encs)
            preds = self._apply_calibration(preds, calibration, last_encs)
            print('[calibration] {}'.format(calibration))
        else:
            val_preds = None
            val_trues = None

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if getattr(self.args, 'save_pred', 1):
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            if val_preds is not None and val_trues is not None:
                np.save(folder_path + 'val_pred.npy', val_preds)
                np.save(folder_path + 'val_true.npy', val_trues)

        return

    def _apply_fixed_postprocess(self, preds, last_encs):
        if getattr(self.args, 'pred_smooth_method', 'none') != 'none':
            preds = self._smooth_predictions(preds)
        if getattr(self.args, 'endpoint_lerp_strength', 0.0) != 0.0:
            preds = self._endpoint_correct_predictions(preds, last_encs)
        return preds

    def _smooth_predictions(self, preds):
        return self._smooth_predictions_with_params(
            preds,
            getattr(self.args, 'pred_smooth_method', 'none'),
            float(getattr(self.args, 'pred_smooth_blend', 1.0)),
            float(getattr(self.args, 'pred_smooth_alpha', 0.1)),
            int(getattr(self.args, 'pred_smooth_window', 3)),
        )

    def _smooth_predictions_with_params(self, preds, method, blend, alpha=0.1, window=3):
        blend = max(0.0, min(1.0, float(blend)))
        if method == 'ema':
            alpha = max(0.0, min(1.0, float(alpha)))
            smoothed = preds.copy()
            for t in range(1, preds.shape[1]):
                smoothed[:, t, :] = alpha * preds[:, t, :] + (1 - alpha) * smoothed[:, t - 1, :]
        elif method == 'ma':
            window = int(window)
            if window <= 1:
                return preds
            left = (window - 1) // 2
            right = window - 1 - left
            padded = np.pad(preds, ((0, 0), (left, right), (0, 0)), mode='edge')
            cumsum = np.cumsum(padded, axis=1)
            cumsum = np.concatenate([np.zeros_like(cumsum[:, :1, :]), cumsum], axis=1)
            smoothed = (cumsum[:, window:, :] - cumsum[:, :-window, :]) / window
        elif method == 'endpoint_linear':
            horizon = preds.shape[1]
            weight = np.linspace(0, 1, horizon, dtype=preds.dtype)[None, :, None]
            smoothed = preds[:, :1, :] * (1 - weight) + preds[:, -1:, :] * weight
        else:
            return preds
        return (1 - blend) * preds + blend * smoothed

    def _endpoint_correct_predictions(self, preds, last_encs):
        strength = float(getattr(self.args, 'endpoint_lerp_strength', 0.0))
        return self._endpoint_correct_predictions_with_strength(preds, last_encs, strength)

    def _endpoint_correct_predictions_with_strength(self, preds, last_encs, strength):
        strength = max(0.0, min(1.0, float(strength)))
        if strength == 0.0:
            return preds
        horizon = preds.shape[1]
        weights = np.linspace(0, strength, horizon, dtype=preds.dtype)[None, :, None]
        anchor = last_encs.reshape(-1, 1, 1).astype(preds.dtype)
        return anchor * (1 - weights) + preds * weights

    def _fit_calibration(self, preds, trues, last_encs):
        method = getattr(self.args, 'calibration_method', 'none')
        if method == 'affine':
            x = preds.reshape(-1).astype(np.float64)
            y = trues.reshape(-1).astype(np.float64)
            var = np.var(x)
            if var < 1e-12:
                scale = 1.0
                bias = 0.0
            else:
                scale = float(np.mean((x - x.mean()) * (y - y.mean())) / var)
                bias = float(y.mean() - scale * x.mean())
            calibrated = scale * preds + bias
            return {'method': method, 'scale': scale, 'bias': bias, 'val_mse': float(np.mean((calibrated - trues) ** 2))}
        if method == 'endpoint_lerp':
            strengths = np.linspace(0.0, float(getattr(self.args, 'calibration_max_strength', 1.0)), int(getattr(self.args, 'calibration_grid_size', 51)))
            best = None
            for strength in strengths:
                calibrated = self._endpoint_correct_predictions_with_strength(preds, last_encs, strength)
                mse = float(np.mean((calibrated - trues) ** 2))
                if best is None or mse < best['val_mse']:
                    best = {'method': method, 'strength': float(strength), 'val_mse': mse}
            return best
        if method == 'smooth_blend':
            smooth_method = getattr(self.args, 'calibration_smooth_method', 'ema')
            alpha = float(getattr(self.args, 'calibration_smooth_alpha', 0.1))
            window = int(getattr(self.args, 'calibration_smooth_window', 3))
            blends = np.linspace(0.0, 1.0, int(getattr(self.args, 'calibration_grid_size', 51)))
            best = None
            for blend in blends:
                calibrated = self._smooth_predictions_with_params(preds, smooth_method, blend, alpha, window)
                mse = float(np.mean((calibrated - trues) ** 2))
                if best is None or mse < best['val_mse']:
                    best = {
                        'method': method,
                        'smooth_method': smooth_method,
                        'blend': float(blend),
                        'alpha': alpha,
                        'window': window,
                        'val_mse': mse,
                    }
            return best
        return {'method': 'none', 'val_mse': float(np.mean((preds - trues) ** 2))}

    def _apply_calibration(self, preds, calibration, last_encs):
        method = calibration.get('method', 'none')
        if method == 'affine':
            return calibration['scale'] * preds + calibration['bias']
        if method == 'endpoint_lerp':
            return self._endpoint_correct_predictions_with_strength(preds, last_encs, calibration['strength'])
        if method == 'smooth_blend':
            return self._smooth_predictions_with_params(
                preds,
                calibration['smooth_method'],
                calibration['blend'],
                calibration['alpha'],
                calibration['window'],
            )
        return preds

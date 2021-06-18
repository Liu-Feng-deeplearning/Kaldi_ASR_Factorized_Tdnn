#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class SOrthConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, padding_mode='zeros'):
    """ Conv1d with a method for stepping towards semi-orthongonality

    http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
    """
    super(SOrthConv, self).__init__()
    kwargs = {'bias': False}
    self.conv = nn.Conv1d(in_channels, out_channels,
                          kernel_size, stride=stride,
                          padding=padding, dilation=dilation,
                          bias=False, padding_mode=padding_mode)
    self.reset_parameters()
    return

  def forward(self, x):
    x = self.conv(x)
    return x

  def step_semi_orth(self):
    with torch.no_grad():
      M = self.get_semi_orth_weight(self.conv)
      self.conv.weight.copy_(M)

  def reset_parameters(self):
    # Standard dev of M init values is inverse of sqrt of num cols
    nn.init._no_grad_normal_(self.conv.weight, 0.,
                             self.get_M_shape(self.conv.weight)[1] ** -0.5)

  def orth_error(self):
    return self.get_semi_orth_error(self.conv).item()

  @staticmethod
  def get_semi_orth_weight(conv1dlayer):
    # updates conv1 weight M using update rule to make it more semi orthogonal
    # based off ConstrainOrthonormalInternal in nnet-utils.cc in Kaldi src/nnet3
    # includes the tweaks related to slowing the update speed
    # only an implementation of the 'floating scale' case
    with torch.no_grad():
      update_speed = 0.125
      orig_shape = conv1dlayer.weight.shape
      # a conv weight differs slightly from TDNN formulation:
      # Conv weight: (out_filters, in_filters, kernel_width)
      # TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
      # the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
      M = conv1dlayer.weight.reshape(
        orig_shape[0], orig_shape[1] * orig_shape[2]).T
      # M now has shape (in_dim[rows], out_dim[cols])
      mshape = M.shape
      if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
        M = M.T
      P = torch.mm(M, M.T)
      PP = torch.mm(P, P.T)
      trace_P = torch.trace(P)
      trace_PP = torch.trace(PP)
      ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

      # the following is the tweak to avoid divergence (more info in Kaldi)
      assert ratio > 0.99
      if ratio > 1.02:
        update_speed *= 0.5
        if ratio > 1.1:
          update_speed *= 0.5

      scale2 = trace_PP / trace_P
      update = P - (torch.matrix_power(P, 0) * scale2)
      alpha = update_speed / scale2
      update = (-4.0 * alpha) * torch.mm(update, M)
      updated = M + update
      # updated has shape (cols, rows) if rows > cols, else has shape (rows, cols)
      # Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
      # Then reshape to (cols, in_filters, kernel_width)
      return updated.reshape(*orig_shape) if mshape[0] > mshape[
        1] else updated.T.reshape(*orig_shape)

  @staticmethod
  def get_M_shape(conv_weight):
    orig_shape = conv_weight.shape
    return (orig_shape[1] * orig_shape[2], orig_shape[0])

  @staticmethod
  def get_semi_orth_error(conv1dlayer):
    with torch.no_grad():
      orig_shape = conv1dlayer.weight.shape
      M = conv1dlayer.weight.reshape(
        orig_shape[0], orig_shape[1] * orig_shape[2]).T
      mshape = M.shape
      if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
        M = M.T
      P = torch.mm(M, M.T)
      PP = torch.mm(P, P.T)
      trace_P = torch.trace(P)
      trace_PP = torch.trace(PP)
      scale2 = torch.sqrt(trace_PP / trace_P) ** 2
      update = P - (torch.matrix_power(P, 0) * scale2)
      return torch.norm(update, p='fro')


class SharedDimScaleDropout(nn.Module):
  def __init__(self, alpha: float = 0.5, dim=1):
    """ Continuous scaled dropout that is const over chosen dim (usually across time)
        Multiplies inputs by random mask taken from Uniform([1 - 2\alpha, 1 + 2\alpha])

    """
    super(SharedDimScaleDropout, self).__init__()
    if alpha > 0.5 or alpha < 0:
      raise ValueError("alpha must be between 0 and 0.5")
    self.alpha = alpha
    self.dim = dim
    self.register_buffer('mask', torch.tensor(0.))

  def forward(self, X):
    if self.training:
      if self.alpha != 0.:
        # sample mask from uniform dist with dim of length 1 in
        # self.dim and then repeat to match size
        tied_mask_shape = list(X.shape)
        tied_mask_shape[self.dim] = 1
        repeats = [1 if i != self.dim else X.shape[self.dim]
                   for i in range(len(X.shape))]
        return X * self.mask.repeat(tied_mask_shape).uniform_(
          1 - 2 * self.alpha, 1 + 2 * self.alpha).repeat(repeats)
        # expected value of dropout mask is 1 so no need to scale
        # outputs like vanilla dropout
    return X


class FTDNNLayer(nn.Module):
  def __init__(self, in_dim, out_dim, bottleneck_dim, context_size=2,
               dilations=(2, 2, 2), paddings=(1, 1, 1), alpha=0.):
    """ 3-stage factorised TDNN

    http://danielpovey.com/files/2018_interspeech_tdnnf.pdf

    """
    super(FTDNNLayer, self).__init__()
    # paddings = [1, 1, 1] if not paddings else paddings
    # dilations = [2, 2, 2] if not dilations else dilations
    assert len(paddings) == 3
    assert len(dilations) == 3
    self.factor1 = SOrthConv(
      in_dim, bottleneck_dim, context_size, padding=paddings[0],
      dilation=dilations[0])
    self.factor2 = SOrthConv(bottleneck_dim, bottleneck_dim,
                             context_size, padding=paddings[1],
                             dilation=dilations[1])
    self.factor3 = nn.Conv1d(bottleneck_dim, out_dim, context_size,
                             padding=paddings[2], dilation=dilations[2],
                             bias=False)
    self._relu = nn.ReLU()
    self.bn = nn.BatchNorm1d(out_dim)
    # self.dropout = SharedDimScaleDropout(alpha=alpha, dim=1)
    self.dropout = nn.Dropout(p=0.5)
    return

  def forward(self, x):
    """ input (batch_size, seq_len, in_dim) """
    assert (x.shape[-1] == self.factor1.conv.weight.shape[1])
    x = self.factor1(x.transpose(1, 2))
    # print("XX1:", x.size())
    x = self.factor2(x)
    # print("XX2:", x.size())
    bottleneck_x = x
    x = self.factor3(x)
    x = self._relu(x)
    x = self.bn(x).transpose(1, 2)
    x = self.dropout(x)
    return x, bottleneck_x.transpose(1, 2)

  def step_semi_orth(self):
    for layer in self.children():
      if isinstance(layer, SOrthConv):
        layer.step_semi_orth()

  def orth_error(self):
    orth_error = 0
    for layer in self.children():
      if isinstance(layer, SOrthConv):
        orth_error += layer.orth_error()
    return orth_error


class FDenseReLU(nn.Module):
  def __init__(self, in_dim, bottle_dim, out_dim):
    super(FDenseReLU, self).__init__()
    self.fc_1 = nn.Linear(in_dim, bottle_dim)
    self.fc_2 = nn.Linear(bottle_dim, out_dim)
    self.bn = nn.BatchNorm1d(out_dim)
    self.nl = nn.ReLU()
    print("init F-Dense layer with  {}/{}".format(bottle_dim, out_dim))

  def forward(self, x):
    x = self.fc_1(x)
    x = self.fc_2(x)
    x = self.nl(x)
    if len(x.shape) > 2:
      x = self.bn(x.transpose(1, 2)).transpose(1, 2)
    else:
      x = self.bn(x)
    return x

  def step_semi_orth(self):
    with torch.no_grad():
      M = self._get_semi_orth_weight(self.fc_1)
      self.fc_1.weight.copy_(M)
      M_2 = self._get_semi_orth_weight(self.fc_2)
      self.fc_2.weight.copy_(M_2)
    return

  @staticmethod
  def _get_semi_orth_weight(fclayer):
    # updates conv1 weight M using update rule to make it more semi orthogonal
    # based off ConstrainOrthonormalInternal in nnet-utils.cc in Kaldi src/nnet3
    # includes the tweaks related to slowing the update speed
    # only an implementation of the 'floating scale' case
    with torch.no_grad():
      update_speed = 0.125
      orig_shape = fclayer.weight.shape
      # a conv weight differs slightly from TDNN formulation:
      # Conv weight: (out_filters, in_filters, kernel_width)
      # TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
      # the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
      M = fclayer.weight.reshape(
        orig_shape[0], orig_shape[1]).T
      # M now has shape (in_dim[rows], out_dim[cols])
      mshape = M.shape
      if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
        M = M.T
      P = torch.mm(M, M.T)
      PP = torch.mm(P, P.T)
      trace_P = torch.trace(P)
      trace_PP = torch.trace(PP)
      ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

      # the following is the tweak to avoid divergence (more info in Kaldi)
      assert ratio > 0.99
      if ratio > 1.02:
        update_speed *= 0.5
        if ratio > 1.1:
          update_speed *= 0.5

      scale2 = trace_PP / trace_P
      update = P - (torch.matrix_power(P, 0) * scale2)
      alpha = update_speed / scale2
      update = (-4.0 * alpha) * torch.mm(update, M)
      updated = M + update
      # updated has shape (cols, rows) if rows > cols, else has shape (rows, cols)
      # Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
      # Then reshape to (cols, in_filters, kernel_width)
      return updated.reshape(*orig_shape) if mshape[0] > mshape[
        1] else updated.T.reshape(*orig_shape)

  @staticmethod
  def _get_semi_orth_error(fc_layer):
    with torch.no_grad():
      orig_shape = fc_layer.weight.shape
      # print("xx:",orig_shape)
      M = fc_layer.weight.reshape(orig_shape[0], orig_shape[1]).T
      mshape = orig_shape
      if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
        M = M.T
      P = torch.mm(M, M.T)
      PP = torch.mm(P, P.T)
      trace_P = torch.trace(P)
      trace_PP = torch.trace(PP)
      scale2 = torch.sqrt(trace_PP / trace_P) ** 2
      update = P - (torch.matrix_power(P, 0) * scale2)
      return torch.norm(update, p='fro')

  def orth_error(self):
    orth_error = self._get_semi_orth_error(
      self.fc_1).item() + self._get_semi_orth_error(self.fc_2).item()
    return orth_error


class TdnnLayer(nn.Module):
  def __init__(
      self,
      input_dim=80,
      output_dim=512,
      context_size=5,
      stride=1,
      dilation=1,
      batch_norm=True,
      dropout_p=0.0,
      padding=0
  ):
    super(TdnnLayer, self).__init__()
    self.context_size = context_size
    self.stride = stride
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.dilation = dilation
    self.dropout_p = dropout_p
    self.padding = padding

    self.kernel = nn.Conv1d(self.input_dim,
                            self.output_dim,
                            self.context_size,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation)

    self.nonlinearity = nn.ReLU()
    self.batch_norm = batch_norm
    if batch_norm:
      self.bn = nn.BatchNorm1d(output_dim)
    self.drop = nn.Dropout(p=self.dropout_p)

  def forward(self, x):
    """
    input: size (batch, seq_len, input_features)
    output: size (batch, new_seq_len, output_features)

    """

    _, _, d = x.shape
    assert (
        d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(
      self.input_dim, d)

    x = self.kernel(x.transpose(1, 2))
    x = self.nonlinearity(x)
    x = self.drop(x)

    if self.batch_norm:
      x = self.bn(x)
    return x.transpose(1, 2)


class TdnnfModel(nn.Module):
  def __init__(self, hp):
    """ the total model architecture is as kaldi/egs/swbd/s5c/local/chain/tuning/run_tdnn_7p and
    kaldi/egs/swbd/s5c/local/chain/tuning/run_tdnn_7q

    The TdnnfLayer is from http://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf
    and more detail about tdnnf-layer can be seen in
    kaldi/egs/swbd/s5c/steps/libs/nnet3/xconfig/composite_layers.py

    and we using skip-connection with (small merge large) in paper.

    """

    super(TdnnfModel, self).__init__()
    in_dim = 80
    output_dim = hp["output_dim"]
    large_dim = hp["large_dim"]
    bottle_dim = hp["bottle_dim"]
    num_tdnnf_layers = hp["num_layers"]

    print("init F-TDNN with {}layer {}/{}".format(
      num_tdnnf_layers, large_dim, bottle_dim))

    self._input_layer = TdnnLayer(input_dim=in_dim,
                                  output_dim=large_dim,
                                  context_size=5, padding=2)
    self._tdnnf_layers = nn.ModuleList([])
    for i in range(num_tdnnf_layers):
      self._tdnnf_layers.append(
        FTDNNLayer(large_dim, large_dim, bottle_dim))

    self._prefinal_layer = FDenseReLU(large_dim, bottle_dim, output_dim)
    self._scale = 0.66
    return

  def forward(self, x):
    """(batch_size, seq_len, in_dim) -> (batch_size, seq_len, out_dim)  """
    x = self._input_layer(x)
    res = None
    for idx, tdnnf_layer in enumerate(self._tdnnf_layers):
      if idx > 0:
        x = x + self._scale * res
      x, _ = tdnnf_layer(x)
      res = x
    x = self._prefinal_layer(x)
    return x

  def model_size(self):
    num = sum(p.numel() for p in self.parameters())
    return num

  def step_ftdnn_layers(self):
    """ The key method to constrain the first two convolutions,
        perform after every SGD step

    """
    for layer in self._tdnnf_layers:
      if isinstance(layer, FTDNNLayer):
        layer.step_semi_orth()
      self._prefinal_layer.step_semi_orth()
    return

  def _set_dropout_alpha(self, alpha):
    # todo: add alpha
    for layer in self.children():
      if isinstance(layer, FTDNNLayer):
        layer.dropout.alpha = alpha
    return

  def get_orth_errors(self):
    """This returns the orth error of the constrained convs, useful for debugging

    """
    count, errors = 0, 0.
    with torch.no_grad():
      for layer in self._tdnnf_layers:
        if isinstance(layer, FTDNNLayer):
          errors += layer.orth_error()
          count += 1
      if self._using_F_Dense:
        errors += self._prefinal_layer.orth_error()
        count += 1
    return errors, count


def __cmd():
  import yaml
  import numpy as np
  device = torch.device('cpu')

  # test for tdnnf_layer
  _inp_features = np.random.random([2, 416, 512])
  _inp_features = torch.from_numpy(_inp_features).float().to(device)
  tdnn_layer = FTDNNLayer(512, 512, 256, context_size=2, dilations=[2, 2, 2],
                          paddings=[1, 1, 1])
  _x, _ = tdnn_layer(_inp_features)
  print(_x.size())

  # test for Tdnnf_model
  _inp_features = np.random.random([2, 416, 80])
  _inp_features = torch.from_numpy(_inp_features).float().to(device)
  with open("hparams.yaml", encoding="utf-8") as yaml_file:
    hp = yaml.safe_load(yaml_file)  # using yaml file for hparames
  tdnnf_model = TdnnfModel(hp["TDNN"])
  print("model size:{:.2f}M".format(tdnnf_model.model_size() / 1000 / 1000))
  output = tdnnf_model(_inp_features)
  print(_inp_features.size(), output.size())

  # test for orth
  err, _ = tdnnf_model.get_orth_errors()
  print("before step err:", err)
  for _ in range(10):
    tdnnf_model.step_ftdnn_layers()
  err, _ = tdnnf_model.get_orth_errors()
  print("after 10step err:", err)
  return


if __name__ == "__main__":
  __cmd()

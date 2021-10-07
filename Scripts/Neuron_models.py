import collections
import tensorflow as tf
import numpy as np
import time
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.layers import base
from tensorflow.python.framework import function
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
from tensorflow.python.ops.gen_array_ops import matrix_diag_part
from tensorflow.python.ops import math_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

@function.Defun()
def derivative(x, current_gradient):
    
    grad = 1. - (tf.nn.tanh(x) * tf.nn.tanh(x))
    return current_gradient * grad

@function.Defun(grad_func=derivative)
def StepFunction(x):
    return tf.nn.relu(tf.sign(x))

def sigmoid_grad(x):
    return math_ops.sigmoid(x) * (1 - math_ops.sigmoid(x))
def tanh_grad(x):
    return 1 - math_ops.tanh(x) * math_ops.tanh(x)
def identity_grad(x):
    return tf.ones_like(x)
def relu_grad(x):
    return tf.where(x >= 0, tf.ones_like(x), tf.zeros_like(x))

_SNUELayerTuple = collections.namedtuple("SNUELayerStateTuple", ("Vm", "h", "dg_t", "dh_t", "ew_in", "ew_in_wo", "ew_rec", "ew_rec_wo", "eb", "eb_wo"))
@tf_export("nn.rnn_cell.SNUELayerStateTuple")
class SNUELayerStateTuple(_SNUELayerTuple):
  __slots__ = ()
  @property
  def dtype(self):
    (Vm, Vm_wog, h, overVth_woh, ew_in, ew_in_wo, ew_rec, ew_rec_wo, eb, eb_wo, ed, ed_wo) = self
    if Vm.dtype != h.dtype or Vm.dtype != Vm_wog.dtype or Vm.dtype != overVth_woh.dtype or Vm.dtype != ew_in.dtype or Vm.dtype != ew_rec.dtype or Vm.dtype != eb.dtype or Vm.dtype != ew_in_wo.dtype or Vm.dtype != ew_rec_wo.dtype or Vm.dtype != eb_wo.dtype:
       raise TypeError("Inconsistent internal state")
    return Vm.dtype

@tf_export("nn.rnn_cell.SNUELayer")
class SNUELayer(LayerRNNCell):
    def __init__(self, num_units, num_units_prev, activation=StepFunction, g = tf.nn.relu, reuse=None, name=None,
                 decay=0.8, trainableDecay=False, initVth=1.0, recurrent=False, initW=None, initH=None):
        super(SNUELayer, self).__init__(_reuse=reuse, name=name)
        self.input_spec = base.InputSpec(ndim=2) # Inputs must be 2-dimensional.
        self._num_units = num_units
        self._num_units_prev = num_units_prev
        self._activation = activation
        if activation == math_ops.tanh:
            self.h_d = tanh_grad
        elif activation == math_ops.sigmoid:
            self.h_d = sigmoid_grad
        elif activation == tf.identity:
            self.h_d = identity_grad
        elif activation == tf.nn.relu:
            self.h_d = relu_grad
        elif activation == StepFunction:
            self.h_d = lambda x: 1. - (tf.nn.tanh(x) * tf.nn.tanh(x))
        else:
            print('Only tanh, sigmoid and identity function are implemented!')
            #raise NotImplementedError('Only tanh, sigmoid and identity function are implemented!')
            self.h_d = None
        
        self.decay = decay
        self.initVth = initVth
        self.trainableDecay = trainableDecay
        self.recurrent = recurrent
        self.g = g
        
        if g == math_ops.tanh:
            self.g_d = tanh_grad
        elif g == math_ops.sigmoid:
            self.g_d = sigmoid_grad
        elif g == tf.identity:
            self.g_d = identity_grad
        elif g == tf.nn.relu:
            self.g_d = relu_grad
        else:
            print('Only tanh, sigmoid and identity function are implemented!')
            self.g_d = None
        
        self.initW = initW
        self.initH = initH
        
        self._state_size = SNUELayerStateTuple(self._num_units, self._num_units, (self._num_units, self._num_units), (self._num_units, self._num_units), 
                                               (self._num_units, self._num_units_prev, self._num_units), (self._num_units, self._num_units_prev, self._num_units),
                                               (self._num_units, self._num_units, self._num_units), (self._num_units, self._num_units, self._num_units),
                                               (self._num_units, self._num_units), (self._num_units, self._num_units)) #Vm, h, ew, eb

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._num_units
    
    def zero_state(self, batch_size, dtype):
        return SNUELayerStateTuple(Vm = tf.zeros((batch_size, self._num_units), dtype=dtype, name='SNUE_Vm'),
                                   h = tf.zeros((batch_size, self._num_units), dtype=dtype, name='SNUE_h'),
                                   dg_t = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='SNUE_dg'),
                                   dh_t = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='SNUE_dh'),
                                   ew_in = tf.zeros(((batch_size, self._num_units, self._num_units_prev, self._num_units)), dtype=dtype, name='SNUE_ew_in'),
                                   ew_in_wo = tf.zeros(((batch_size, self._num_units, self._num_units_prev, self._num_units)), dtype=dtype, name='SNUE_ew_in_wo'),
                                   ew_rec = tf.zeros(((batch_size, self._num_units, self._num_units, self._num_units)), dtype=dtype, name='SNUE_ew_rec'),
                                   ew_rec_wo = tf.zeros(((batch_size, self._num_units, self._num_units, self._num_units)), dtype=dtype, name='SNUE_ew_rec_wo'),
                                   eb = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='SNUE_eb'),
                                   eb_wo = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='SNUE_eb_wo'),
                                   )
        
    def grad(self, grad, state, inp, out, optimizer, apply): 
        op_list = []
        
        for key in self.eligibility_trace_dict:
            el = getattr(state, self.eligibility_trace_dict[key])
            if self._bias.name in key:
                m_grad = tf.einsum('bj,bjl->bl', grad, el)
            else:
                m_grad = tf.einsum('bj,bjkl->bkl', grad, el)
            
            m_grad = tf.reduce_sum(m_grad, 0)
            
            if apply:
                mod_grad = m_grad + self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]]
            else:
                mod_grad = tf.zeros_like(m_grad)
                
            with tf.control_dependencies([mod_grad]):
                if self._kernel.name in key and self._kernel.trainable:
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([mod_grad], [self._kernel]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
                
                if self.recurrent and self._recurrent_kernel.name in key and self._recurrent_kernel.trainable:
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([mod_grad], [self._recurrent_kernel]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
                
                if self._bias.name in key and self._bias.trainable:
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([mod_grad], [self._bias]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
        
        return op_list
    
    def grad_v(self, grad, vars, state, optimizer, apply): 
        op_list = []
        return op_list
    
    def get_grad(self, grad, state, inp, out, apply):
        return_list = []
        var_list = []
        op_list = []
        return return_list, var_list, op_list

    def build(self, inputs_shape):
        #print(inputs_shape)
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
        
        input_weights = inputs_shape[1].value
        
        self.eligibility_trace_dict = {}
        self.eligibility_trace_storage_dict = {}
        
        add_name = ''
        
        if self.initW is None:
            self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME + add_name, shape=[input_weights, self._num_units])
        else:
            self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME + add_name, shape=[input_weights, self._num_units], initializer=tf.constant_initializer(self.initW))
        
        self.eligibility_trace_dict.update({self._kernel.name: 'ew_in'})
        self.el_kernel_storage = self.add_variable(_WEIGHTS_VARIABLE_NAME + add_name + '_storage', shape=[input_weights, self._num_units], initializer=tf.zeros_initializer, trainable=False)
        self.eligibility_trace_storage_dict.update({'ew_in': self.el_kernel_storage})
        
        if self.recurrent:
            if self.initH is None:
                self._recurrent_kernel = self.add_variable('kernel_h' + add_name, shape=[self._num_units, self._num_units])
            else:
                self._recurrent_kernel = self.add_variable('kernel_h' + add_name, shape=[self._num_units, self._num_units], initializer=tf.constant_initializer(self.initH))
            self.eligibility_trace_dict.update({self._recurrent_kernel.name: 'ew_rec'})
            self.el_rec_kernel_storage = self.add_variable('kernel_h' + add_name + '_storage', shape=[self._num_units, self._num_units], initializer=tf.zeros_initializer, trainable=False)
            self.eligibility_trace_storage_dict.update({'ew_rec': self.el_rec_kernel_storage})

        self._bias = self.add_variable(_BIAS_VARIABLE_NAME + add_name, shape=[self._num_units], initializer=tf.constant_initializer(self.initVth, dtype=self.dtype))
        self.eligibility_trace_dict.update({self._bias.name: 'eb'})
        self.el_bias_storage = self.add_variable(_BIAS_VARIABLE_NAME + add_name + '_storage', shape=[self._num_units], initializer=tf.zeros_initializer, trainable=False)
        self.eligibility_trace_storage_dict.update({'eb': self.el_bias_storage})
        
        if self.trainableDecay:
            self._decay = self.add_variable("decay" + add_name, shape=[self._num_units])
            self.eligibility_trace_dict.update({self._decay.name: 'ed'})
            self.el_decay_storage = self.add_variable("decay" + add_name + '_storage', shape=[self._num_units], initializer=tf.zeros_initializer, trainable=False)
            self.eligibility_trace_storage_dict.update({'ed': self.el_decay_storage})
        else:
            self._decay = self.add_variable("decay" + add_name, shape=[self._num_units], initializer=tf.constant_initializer(self.decay, dtype=tf.float32), trainable=False)
    
        self.built = True

    def call(self, inputs, state):
        (Vm, h, dg_t, dh_t, ew_in, ew_in_wo, ew_rec, ew_rec_wo, eb, eb_wo) = state
        
        h_t = h
        Vm_t = Vm
        Vm = tf.multiply(Vm, 1.0-h)
        Vm = tf.multiply(Vm, self._decay)
        Vm = tf.add(tf.matmul(inputs, self._kernel), Vm)
        
        if self.recurrent:
            Vm = tf.add(tf.matmul(h, self._recurrent_kernel), Vm)
            
        Vm_wog = Vm
        Vm = self.g(Vm_wog)
        
        if self.g_d != None:
            dg = tf.stop_gradient(tf.matrix_diag(self.g_d(Vm_wog)))
        else:
            dg = tf.stop_gradient(batch_jacobian(Vm, Vm_wog))
        
        overVth = Vm - self._bias
        out = self._activation(overVth)
        out.set_shape(overVth.shape)
        
        if self.h_d != None:
            dh = tf.stop_gradient(tf.matrix_diag(self.h_d(overVth)))
        else:
            dh = tf.stop_gradient(batch_jacobian(out, overVth))
        
        if self.recurrent:
            '''
            pdst_pdytm1 = tf.stop_gradient(batch_jacobian(Vm_new, h_t))
            dst_dstm1 = tf.stop_gradient(batch_jacobian(Vm_new, Vm) + tf.einsum('bjk,bkl->bjl', pdst_pdytm1, dh_t))
            
            #dst_dstm1 = tf.stop_gradient(batch_jacobian(Vm_t * (1.0-h) * self._decay, Vm_t))
            pdyt_pdst = tf.stop_gradient(batch_jacobian(out, Vm_new))
            #pdst_pdytm1 = tf.stop_gradient(batch_jacobian(self.g(tf.stop_gradient(Vm_t) * (1.0 - h_t) * self._decay + tf.matmul(inputs, self._kernel) + tf.matmul(h, self._recurrent_kernel)), h_t))
            #pdst_pdytm1 = tf.stop_gradient(batch_jacobian(Vm_new, h))
            #pdst_pdth = tf.stop_gradient(jacobian(c, self._bias))
            new_ytth = tf.stop_gradient(jacobian(self._activation(tf.stop_gradient(Vm_new) - self._bias), self._bias))
            
            new_eb_wo = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dst_dstm1, eb_wo) \
               #+ linalg.einsum('bjk,bklm->bjlm', pdst_pdytm1_pdytm1_pdstm1, ep_w)\
               #+ pdst_pdth\
               #+ pdst_pdytm1_pdth\
               + tf.einsum('bjk,bkl->bjl', pdst_pdytm1, -dh_t)
               )
            fpart = dst_dstm1
            new_eb = tf.stop_gradient(tf.einsum('bij,bjk->bik', pdyt_pdst, new_eb_wo) + new_ytth)
            '''
            #Bias
            fpart = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dg, (tf.einsum('bjk,kl->blj', dh_t, self._recurrent_kernel) +\
                    (tf.expand_dims(self._decay * (1 - h_t), 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0)
                      - tf.expand_dims(self._decay * Vm_t, 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) * dh_t))))
            
            spart = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dg, (tf.einsum('bjk,kl->blj', -dh_t, self._recurrent_kernel) +\
                     (tf.expand_dims(self._decay * Vm_t, 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) * dh_t)))
                                                                 )
            new_eb_wo = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', fpart, eb_wo) + spart)
            new_eb = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dh, new_eb_wo) - dh)
            
            #Input weights
            a = tf.stop_gradient(tf.expand_dims(tf.expand_dims(tf.eye(self._num_units, self._num_units), 1), 0) * tf.expand_dims(tf.expand_dims(inputs, 1), 3))
            spart = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dg, a))
            
            new_ew_in_wo = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', fpart, ew_in_wo) + spart)
            new_ew_in = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dh, new_ew_in_wo))
            
            #Recurrent weights
            a = tf.stop_gradient(tf.expand_dims(tf.expand_dims(tf.eye(self._num_units, self._num_units), 1), 0) * tf.expand_dims(tf.expand_dims(h_t, 1), 3))
            spart = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dg, a))
            
            new_ew_rec_wo = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', fpart, ew_rec_wo) + spart)
            new_ew_rec = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dh, new_ew_rec_wo))
                      
        else:
            fpart = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dg, ((tf.expand_dims(self._decay * (1 - h_t), 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) - tf.expand_dims(self._decay * Vm_t, 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) * dh_t))))
            
            spart = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dg, ((tf.expand_dims(self._decay * Vm_t, 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) * dh_t))))
            
            new_eb_wo = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', fpart, eb_wo) + spart)
            new_eb = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dh, new_eb_wo) - dh)
            
            #Input weights
            fpart = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dg, ((tf.expand_dims(self._decay * (1 - h_t), 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) - tf.expand_dims(self._decay * Vm_t, 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) * dh_t))))
            a = tf.stop_gradient(tf.expand_dims(tf.expand_dims(tf.eye(self._num_units, self._num_units), 1), 0) * tf.expand_dims(tf.expand_dims(inputs, 1), 3))
            spart = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dg, a))
            
            new_ew_in_wo = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', fpart, ew_in_wo) + spart)
            new_ew_in = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dh, new_ew_in_wo))
            
            #Recurrent weights
            new_ew_rec_wo = tf.stop_gradient(ew_rec_wo)
            new_ew_rec = tf.stop_gradient(ew_rec)
        
        new_state = SNUELayerStateTuple(Vm, out, dg, dh, new_ew_in, new_ew_in_wo, new_ew_rec, new_ew_rec_wo, new_eb, new_eb_wo)
        
        return out, new_state
    
_DenseTuple = collections.namedtuple("DenseTuple", ("ew", "ew_wo", "eb", "eb_wo"))
@tf_export("nn.rnn_cell.DenseTuple")
class DenseTuple(_DenseTuple):
  __slots__ = ()
  @property
  def dtype(self):
    (ew, ew_wo, eb, eb_wo) = self
    if ew.dtype != ew_wo.dtype or ew.dtype != eb.dtype or ew.dtype != eb_wo.dtype:
       raise TypeError("Inconsistent internal state: %s vs %s vs %s vs %s vs %s" % (str(ew.dtype), str(ew_wo.dtype), str(eb.dtype), str(eb_wo.dtype)))
    return ew.dtype
    
class DenseRNNLayer(LayerRNNCell):
    def __init__(self, units, num_units_prev, activation=tf.identity, reuse=None, name=None, initW=None, initB=None):
        
        super(DenseRNNLayer, self).__init__(_reuse=reuse, name=name)
        self.input_spec = base.InputSpec(ndim=2)
        self._num_units = units
        self._num_units_prev = num_units_prev
        self.initW = initW
        self.initB = initB
        self.activation = tf.identity
            
        self._state_size = DenseTuple((self._num_units_prev, self._num_units), (self._num_units_prev, self._num_units),
                                      self._num_units, self._num_units)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._num_units
    
    def zero_state(self, batch_size, dtype):
        return DenseTuple(ew = tf.zeros(((batch_size, self._num_units, self._num_units_prev, self._num_units)), dtype=dtype, name='Dense_ew'),
                          ew_wo = tf.zeros(((batch_size, self._num_units, self._num_units_prev, self._num_units)), dtype=dtype, name='Dense_ew_wo'),
                          eb = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='Dense_eb'),
                          eb_wo = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='Dense_eb_wo'),
                          )
        
    def grad(self, grad, state, inp, out, optimizer, apply): 
        op_list = []
        
        for key in self.eligibility_trace_dict:
            el = getattr(state, self.eligibility_trace_dict[key])
            if self._bias.name in key:
                m_grad = tf.einsum('bj,bjk->bk', grad, el)
            else:
                m_grad = tf.einsum('bj,bjkl->bkl', grad, el)
                
            m_grad = tf.reduce_sum(m_grad, 0)
                    
            if apply:
                mod_grad = m_grad + self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]]
            else:
                mod_grad = tf.zeros_like(m_grad)
                
            with tf.control_dependencies([mod_grad]):
                if self._kernel.name in key and self._kernel.trainable:
                    
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([mod_grad], [self._kernel]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
                
                if self._bias.name in key and self._bias.trainable:
                    
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([mod_grad], [self._bias]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
        
        return op_list
    
    def grad_v(self, grad, vars, state, optimizer, apply): 
        op_list = []
        return op_list
    
    def get_grad(self, grad, state, inp, out, apply): 
        return_list = []
        var_list = []
        op_list = []
        
        for key in self.eligibility_trace_dict:
            el = getattr(state, self.eligibility_trace_dict[key])
            if self._bias.name in key:
                m_grad = tf.einsum('bj,bjk->bk', grad, el)
            else:
                m_grad = tf.einsum('bj,bjkl->bkl', grad, el)
                
            m_grad = tf.reduce_sum(m_grad, 0)
                    
            if apply:
                mod_grad = m_grad + self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]]
            else:
                mod_grad = tf.zeros_like(m_grad)
                
            with tf.control_dependencies([mod_grad]):
                if self._kernel.name in key and self._kernel.trainable:
                    return_list.append(mod_grad)
                    var_list.append(self._kernel)
                    
                    if apply:
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
                
                if self._bias.name in key and self._bias.trainable:
                    return_list.append(mod_grad)
                    var_list.append(self._bias)
                    
                    if apply:
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
                    
        with tf.control_dependencies(op_list):
            return return_list, var_list, op_list

    def build(self, inputs_shape):
        
        self.eligibility_trace_dict = {}
        self.eligibility_trace_storage_dict = {}
        input_weights = inputs_shape[1].value
        
        add_name = ''
        
        if type(self.initW) != np.ndarray:
            self._kernel = self.add_variable('dense/kernel' + add_name, shape=[input_weights, self._num_units], initializer=tf.constant_initializer(np.random.normal(size=[input_weights, self._num_units])))
        else:
            self._kernel = self.add_variable('dense/kernel' + add_name, shape=[input_weights, self._num_units], initializer=tf.constant_initializer(self.initW))
        self.eligibility_trace_dict.update({self._kernel.name: 'ew'})
        self.el_kernel_storage = self.add_variable('dense/kernel_storage' + add_name, shape=[input_weights, self._num_units], initializer=tf.zeros_initializer, trainable=False)
        self.eligibility_trace_storage_dict.update({'ew': self.el_kernel_storage})
        
        if type(self.initB) != np.ndarray:
            self._bias = self.add_variable('dense/bias' + add_name, shape=[self._num_units], initializer=tf.constant_initializer(np.random.normal(size=[self._num_units])))
        else:
            self._bias = self.add_variable('dense/bias' + add_name, shape=[self._num_units], initializer=tf.constant_initializer(self.initB))#, initializer=tf.constant_initializer(self.initB))
        self.eligibility_trace_dict.update({self._bias.name: 'eb'})
        self.el_bias_storage = self.add_variable('dense/bias_storage' + add_name, shape=[self._num_units], initializer=tf.zeros_initializer, trainable=False)
        self.eligibility_trace_storage_dict.update({'eb': self.el_bias_storage})
        
        self.built = True

    def call(self, inputs, state):
        ew, ew_wo, eb, eb_wo = state
        
        bias = tf.tile(tf.expand_dims(self._bias, 0), (tf.shape(inputs)[0], 1))
        kernel = tf.tile(tf.expand_dims(self._kernel, 0), (tf.shape(inputs)[0], 1, 1))
        
        out_woh = tf.einsum('bi,bij->bj', inputs, kernel) + bias
        
        out = self.activation(out_woh)
        
        dh = batch_jacobian(out, out_woh)
        
        #Bias
        spart = tf.tile(tf.expand_dims(tf.eye(self._num_units, self._num_units), 0), (tf.shape(out)[0], 1, 1))
        
        new_eb_wo = eb_wo
        new_eb = batch_jacobian(out, bias)
            
        #Input weights
        spart = tf.einsum('il,bk->bikl', tf.eye(self._num_units, self._num_units), tf.einsum('kj,bj->bk', tf.eye(self._num_units_prev, self._num_units_prev), inputs))
        
        new_ew_wo = spart#tf.ones_like(spart)
        new_ew = batch_jacobian(out, kernel)
        
        return out, DenseTuple(new_ew, new_ew_wo, new_eb, new_eb_wo)
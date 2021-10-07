from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest
from tensorflow.python.ops.array_ops import zeros, zeros_like
from tensorflow.python.ops.gen_control_flow_ops import no_op
from tensorflow.compat.v1.nn.rnn_cell import MultiRNNCell as MultiRNNCellBase
#from tensorflow.python.rnn_cell import _RefVariableProcessor, _DenseResourceVariableProcessor

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# This can be used with self.assertRaisesRegexp for assert_like_rnncell.
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"

class MultiRNNCell(MultiRNNCellBase):
  
  def __init__(self, cells, state_is_tuple=True):
    super(MultiRNNCell, self).__init__(cells, state_is_tuple)
    self.old_outputs = []
    self.old_states = []
    self.new_outputs = []

  def call(self, inputs, state):
    cur_state_pos = 0
    cur_inp = inputs
    new_states = []
    old_states = []
    old_outputs = self.new_outputs
    first_round = self.new_outputs == [] and True or False
    inp = []
    old_outputs = []
    self.new_outputs = []
    
    inp.append(inputs)
            
    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_%d" % i):
        if self._state_is_tuple:
          if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s" %
                (len(self.state_size), state))
          cur_state = state[i]
        else:
          cur_state = array_ops.slice(state, [0, cur_state_pos],
                                      [-1, cell.state_size])
          cur_state_pos += cell.state_size
        
        cur_inp, new_state = cell(cur_inp, cur_state)
        
        if i < (len(self._cells) - 1):
            inp.append(cur_inp)
        
        old_states.append(cur_state)        
        self.new_outputs.append(cur_inp)
        new_states.append(new_state)

    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))
    
    if old_outputs == []:
        old_outputs = [zeros_like(out) for out in self.new_outputs[:-1]]

    return cur_inp, new_states, self.new_outputs, old_states, old_outputs, inp

  def grad(self, grads, state, apply=True):
    op_list = []
    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_%d" % i):
        op_list.extend(cell.grad(grads, state[i], apply))
    return op_list
    
  def grad_c(self, grads, state, inp, new_outputs, cell_ind, optimizer, apply=True):
    op_list = []
    cell = self._cells[cell_ind]
    with vs.variable_scope("cell_%d" % cell_ind):
      op_list.extend(cell.grad(grads, state[cell_ind], inp[cell_ind], new_outputs[cell_ind], optimizer, apply))
    return op_list

  def get_grad_c(self, grads, state, inp, new_outputs, cell_ind, apply=True):
    g_list = []
    v_list = []
    cell = self._cells[cell_ind]
    with vs.variable_scope("cell_%d" % cell_ind):
      g, v, _ = cell.get_grad(grads, state[cell_ind], inp[cell_ind], new_outputs[cell_ind], apply)
      g_list.extend(g)
      v_list.extend(v)
    return g_list, v_list

  def grad_c_v(self, grads, vars, state, cell_ind, optimizer, apply=True):
    op_list = []
    cell = self._cells[cell_ind]
    with vs.variable_scope("cell_%d" % cell_ind):
      op_list.extend(cell.grad_v(grads, vars, state[cell_ind], optimizer, apply))
    return op_list
    
  def store_bptt_grad(self, grads, cell_ind, time):
    op_list = [no_op()]
    cell = self._cells[cell_ind]
    op_list.extend(cell.add_bptt(grads))
    return op_list
    
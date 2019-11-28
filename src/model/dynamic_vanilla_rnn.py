# encoding=utf-8-sig
# 注释日期2018年12月7日
# 参考版本 Tensorflow 1.12.0 dynamic_rnn函数
# 删除了大量校验代码，降低鲁棒性以换取可读性
# 删除了大量文档
# 禁止输入keras cell
# 取消了对eager execution的支持
# 取消了对Nested Tuple的支持
# 取消了skip conditional的跳转设计
# 强制要求输入三维Time Major数据
# 强制要求输入sequence_length
# 强制要求输入initial_state
# 强制输入中的batch_size置None
# 将代码从538行删减到了226行

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
import autoencoder
import numpy as np
from rnn_cell import GRUCell, LSTMCell, RawCell

# 拼接矩阵的shape
_concat = rnn_cell_impl._concat


def _maybe_tensor_shape_from_tensor(shape):
    if isinstance(shape, ops.Tensor):
        return tensor_shape.as_shape(tensor_util.constant_value(shape))
    else:
        return shape


def _should_cache():
    """Returns True if a default caching device should be set, otherwise False."""
    if context.executing_eagerly():
        return False
    # Don't set a caching device when running in a loop, since it is possible that
    # train steps could be wrapped in a tf.while_loop. In that scenario caching
    # prevents forward computations in loop iterations from re-reading the
    # updated weights.
    ctxt = ops.get_default_graph()._get_control_flow_context()
    # pylint: disable=protected-access
    return control_flow_util.GetContainingWhileContext(ctxt) is None


def _vanilla_dynamic_rnn(cell, inputs, sequence_length, initial_state, parallel_iterations=32, swap_memory=False,
                         scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    Args:
      cell: An instance of RNNCell.
      inputs: The RNN inputs.
        Time Major Tensor with Shape [batch_size, max_time, Depth]
      sequence_length:  An int32/int64 vector sized `[batch_size]`
      initial_state: (optional) An initial state for the RNN.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      parallel_iterations: same as official explanation
      swap_memory: same as official explanation
      scope:same as official explanation

    Returns:
      A pair (outputs, state) where:

      outputs: The RNN output `Tensor`. with shape [Time, Batch_Size, Num_output]

      state: The final state.  with shape, [batch_size, num_state]

    Raises:
      TypeError: If `cell` is not an instance of RNNCell.
      ValueError: If inputs is None or an empty list.
    """

    with vs.variable_scope(scope or "rnn") as varscope:
        # Create a new scope in which the caching device is either
        # determined by the parent scope, or is set to place the cached
        # Variable using the same placement as for the rest of the RNN.
        if _should_cache():
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        # Just to find it in the graph.
        sequence_length = array_ops.identity(sequence_length, name="sequence_length")

        state = initial_state

        # Construct an initial output
        input_shape = array_ops.shape(inputs)
        time_steps, batch_size = input_shape[0], input_shape[1]
        const_time_step, const_batch_size = inputs.get_shape().as_list()[:2]

        # Prepare dynamic conditional copying of state & output
        def _create_zero_arrays(size):
            size = _concat(batch_size, size)
            return array_ops.zeros(array_ops.stack(size), state.dtype)

        zero_output = _create_zero_arrays(cell.output_size)
        time = array_ops.constant(0, dtype=dtypes.int32, name="time")

        with ops.name_scope("vanilla") as scope:
            base_name = scope

            # TensorArray，为了while_loop所准备的特定数据结构
            def _create_ta(name, element_shape, dtype):
                return tensor_array_ops.TensorArray(dtype=dtype,
                                                    size=time_steps,
                                                    element_shape=element_shape,
                                                    tensor_array_name=base_name + name)

            output_ta = _create_ta("output",
                                   element_shape=(tensor_shape.TensorShape([const_batch_size])
                                                  .concatenate(_maybe_tensor_shape_from_tensor(cell.output_size))),
                                   dtype=state.dtype)
            input_ta = _create_ta("input", element_shape=inputs.shape[1:], dtype=inputs.dtype)
            # input_ta 赋值
            input_ta = input_ta.unstack(inputs)

            def _time_step(time_, output_ta_t, state_):
                """Take a time step of the dynamic RNN.

                Args:
                  time_: int32 scalar Tensor.
                  output_ta_t: A TensorArray that represent the output.

                Returns:
                  The tuple (time + 1, output_ta_t with updated flow, new_state).
                """

                # 读取数据
                input_t = input_ta.read(time_)
                input_t.set_shape(inputs.get_shape().as_list()[1:])
                call_cell = lambda: cell(input_t, state_)

                output_, new_state = _rnn_step(
                    time=time_,
                    sequence_length=sequence_length,
                    zero_output=zero_output,
                    state=state_,
                    call_cell=call_cell)

                output_ta_t = output_ta_t.write(time_, output_)
                return time_ + 1, output_ta_t, new_state

            # Make sure that we run at least 1 step, if necessary, to ensure
            # the TensorArrays pick up the dynamic shape.
            loop_bound = time_steps

            _, output_final_ta, final_state = control_flow_ops.while_loop(
                cond=lambda time_, *_: time_ < loop_bound,
                body=_time_step,
                loop_vars=(time, output_ta, state),
                parallel_iterations=parallel_iterations,
                maximum_iterations=time_steps,
                swap_memory=swap_memory)

            output_final = output_final_ta.stack()
            shape = _concat([const_time_step, const_batch_size], cell.output_size, static=True)
            output_final.set_shape(shape)
        return output_final, final_state


def _rnn_step(time, sequence_length, zero_output, state, call_cell):
    """
    Calculate one step of a dynamic RNN minibatch.
    Args: same as official design
    Returns:
      A tuple of (`final_output`, `final_state`)
        final_output is a `Tensor` matrix of shape [batch_size, output_size]
        final_state is either a single `Tensor` matrix
    Raises:
      ValueError: If the cell returns a state tuple whose length does not match
        that returned by `state_size`.
    """

    # Convert state to a list for ease of use
    # 由于LSTM本身的state就是一个tuple，因此此处保留了flatten
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)

    # Vector describing which batch entries are finished.
    copy_cond = time >= sequence_length

    def _copy_one_through(output_, new_output_):
        # TensorArray and scalar get passed through.
        if isinstance(output_, tensor_array_ops.TensorArray):
            return new_output_
        if output_.shape.ndims == 0:
            return new_output_
        # Otherwise propagate the old or the new value.
        with ops.colocate_with(new_output_):
            return array_ops.where(copy_cond, output_, new_output_)

    def _copy_some_through(flat_new_output, flat_new_state):
        flat_new_output = [_copy_one_through(zero_output_, new_output_)
                           for zero_output_, new_output_ in zip(flat_zero_output, flat_new_output)]
        flat_new_state = [_copy_one_through(state_, new_state_)
                          for state_, new_state_ in zip(flat_state, flat_new_state)]
        return flat_new_output + flat_new_state

    new_output, new_state = call_cell()
    nest.assert_same_structure(state, new_state)
    new_state = nest.flatten(new_state)
    new_output = nest.flatten(new_output)
    final_output_and_state = _copy_some_through(new_output, new_state)

    if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
        raise ValueError("Internal error: state and output were not concatenated "
                         "correctly.")
    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]

    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for sub_state, flat_sub_state in zip(final_state, flat_state):
        if not isinstance(sub_state, tensor_array_ops.TensorArray):
            sub_state.set_shape(flat_sub_state.get_shape())

    final_output = nest.pack_sequence_as(
        structure=zero_output, flat_sequence=final_output)
    final_state = nest.pack_sequence_as(
        structure=state, flat_sequence=final_state)

    return final_output, final_state


def vanilla_rnn_model(cell, num_steps, num_hidden, num_context, num_event, keep_rate_input, dae_weight,
                      phase_indicator, embedded_size, autoencoder_initializer=tf.initializers.orthogonal()):
    """
    标准输入规定为TBD
    :param cell:
    :param num_steps:
    :param num_hidden:
    :param num_context: 要求Context变量全部变为二值变量
    :param num_event:
    :param dae_weight:
    :param keep_rate_input:
    :param phase_indicator:
    :param embedded_size: 大于0时执行对输入的自编码，值即为最终降到的维度
    :param autoencoder_initializer:
    :return:
    loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    with tf.name_scope('vanilla_model'):
        with tf.name_scope('data_source'):
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            event_placeholder = tf.placeholder(tf.float32, [num_steps, None, num_event], name='event_placeholder')
            context_placeholder = tf.placeholder(tf.float32, [num_steps, None, num_context], name='context_placeholder')
            sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
            y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')
            initial_state = cell.get_initial_state(batch_size)

        with tf.name_scope('autoencoder'):
            # input_x 用于计算重构原始向量时产生的误差
            processed_input, autoencoder_weight = autoencoder.denoising_autoencoder(
                phase_indicator, context_placeholder, keep_rate_input, embedded_size, autoencoder_initializer)

        with tf.name_scope('vanilla_rnn'):
            input_ = tf.concat([processed_input, event_placeholder], axis=2)
            outputs, final_state = _vanilla_dynamic_rnn(cell, input_, sequence_length, initial_state)

            # 在使用时LSTM时比较麻烦，因为state里同时包含了hidden state和cell state，只有后者是需要输出的
            # 因此需要额外需要做一个split。这种写法非常不优雅，但是我想了一想，也没什么更好的办法
            # 做split时，需要特别注意一下state里到底谁前谁后
            output_length = outputs.shape[2].value
            state_length = final_state.shape[1].value
            if output_length == state_length:
                # 不需要做任何事情
                pass
            elif output_length * 2 == state_length:
                final_state = tf.split(final_state, 2, axis=1)[0]
            else:
                raise ValueError('Invalid Size')

    with tf.variable_scope('output_layer'):
        output_weight = tf.get_variable("weight", [num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])

    with tf.name_scope('loss'):
        unnormalized_prediction = tf.matmul(final_state, output_weight) + bias
        loss_pred = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)

        if embedded_size > 0:
            loss_dae = autoencoder.autoencoder_loss(embedding=processed_input, origin_input=context_placeholder,
                                                    weight=autoencoder_weight)
        else:
            loss_dae = 0

        loss = loss_pred + loss_dae * dae_weight

    with tf.name_scope('prediction'):
        prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
        sequence_length, final_state


def unit_test():
    num_hidden = 10
    num_steps = 20
    keep_prob = 1.0
    num_context = 50
    num_event = 25
    keep_rate_input = 0.8
    dae_weight = 1
    embedded_size = 15

    if embedded_size > 0:
        input_length = embedded_size
    else:
        input_length = num_event+num_context

    initializer_o = tf.initializers.orthogonal()
    initializer_z = tf.initializers.zeros()
    phase_indicator = tf.placeholder(tf.int16, [])

    # 试验阶段
    test_cell_type = 2
    if test_cell_type == 0:
        a_cell = GRUCell(num_hidden=num_hidden, input_length=input_length, weight_initializer=initializer_o,
                         bias_initializer=initializer_z, keep_prob=keep_prob, phase_indicator=phase_indicator,
                         name='gru')
        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            sequence_length, final_state = vanilla_rnn_model(a_cell, num_steps, num_hidden, num_context, num_event,
                                                             keep_rate_input, dae_weight, phase_indicator,
                                                             embedded_size)
    elif test_cell_type == 1:
        b_cell = RawCell(num_hidden=num_hidden,  weight_initializer=initializer_o, bias_initializer=initializer_z,
                         keep_prob=keep_prob, input_length=input_length, phase_indicator=phase_indicator,
                         name='raw')
        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            sequence_length, final_state = vanilla_rnn_model(b_cell, num_steps, num_hidden, num_context, num_event,
                                                             keep_rate_input, dae_weight, phase_indicator,
                                                             embedded_size)

    elif test_cell_type == 2:
        c_cell = LSTMCell(num_hidden=num_hidden, input_length=input_length, weight_initializer=initializer_o,
                          bias_initializer=initializer_z, keep_prob=keep_prob, phase_indicator=phase_indicator,
                          name='lstm')
        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            sequence_length, final_state = vanilla_rnn_model(c_cell, num_steps, num_hidden, num_context, num_event,
                                                             keep_rate_input, dae_weight, phase_indicator,
                                                             embedded_size)
    else:
        raise ValueError('')
    print('finish')

    batch_size_value = 32
    event = np.random.normal(0, 1, [num_steps, batch_size_value, num_event])
    context_ = np.random.normal(0, 1, [num_steps, batch_size_value, num_context])

    sequence_length_value = np.random.randint(1, 8, [batch_size_value])
    feed_dict = {event_placeholder: event, context_placeholder: context_, batch_size: batch_size_value,
                 phase_indicator: 1, sequence_length: sequence_length_value}

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pred = sess.run(prediction, feed_dict=feed_dict)
        print(pred)


if __name__ == '__main__':
    unit_test()

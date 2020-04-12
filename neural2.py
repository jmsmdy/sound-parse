import tensorflow as tf
import numpy as np
import os

scaling=0.01

class STFTLayer(tf.Module):
    def __init__(self, input_features, frame_length, frame_step, name=None):
        super(STFTLayer, self).__init__(name=name)
        self.scaling = tf.constant(1.0, dtype=tf.float32)
        self.scalings = []
        self.input_features = input_features
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_frames = -(-input_features // frame_step)
        i = 0
        while (2 ** i) < frame_length:
            i += 1
        self.fft_length = 2 ** i
        self.fft_unique_bins = self.fft_length // 2 + 1
        self.output_features = 2 * self.fft_unique_bins * self.num_frames
        print(f'STFT Layer. Input Features: {self.input_features} Output Features: {self.output_features}')
        print(f'FFT Unique Bins: {self.fft_unique_bins} Num Frames: {self.num_frames}')
        
    def reset(self):
        pass
        
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    @tf.function
    def __call__(self, x): 
        stft_shape = x.shape
        stft = tf.signal.stft(x,
                              self.frame_length,
                              self.frame_step,
                              fft_length=self.fft_length,
                              window_fn=tf.signal.hann_window,
                              pad_end=True,
                              name=None)
        return tf.concat(tf.unstack(tf.concat([tf.math.real(stft), tf.math.imag(stft)], axis=-2), axis=-2), axis=-1)
    
class RFFTLayer(tf.Module):
    def __init__(self, input_features, name=None):
        super(RFFTLayer, self).__init__(name=name)
        self.scaling = tf.constant(1.0, dtype=tf.float32)
        self.scalings = []
        self.input_features = input_features
        i = 0
        while (2 ** i) < self.input_features:
            i += 1
        self.fft_length = [2 ** i]
        self.output_features = 2 * (1 + 2 ** (i-1))
        print(f'RFFT Layer. Input Features: {self.input_features} Output Features: {self.output_features}')
        
    def reset(self):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    @tf.function
    def __call__(self, x):
        rfft = tf.signal.rfft(x, fft_length=self.fft_length)
        return tf.concat([tf.math.real(rfft), tf.math.imag(rfft)], axis=-1)

        return result
    
class ReluLayer(tf.Module):
    def __init__(self, input_features, output_features, scaling=None, name=None):
        super(ReluLayer, self).__init__(name=name)
        self.input_features = input_features
        self.output_features = output_features
        if scaling:
            self.scaling = tf.constant(scaling, dtype=tf.float32)
        else:
            self.scaling = tf.constant(np.sqrt(2 / self.input_features), dtype=tf.float32)
        self.w = tf.Variable(tf.random.normal([self.input_features, self.output_features],
                                              dtype=tf.float32,
                                              mean=0.0,
                                              stddev=self.scaling), name='w')
        self.b = tf.Variable(tf.zeros([self.output_features], dtype=tf.float32), name='b')
        print(f'RELU Layer. Input Features: {self.input_features} Output Features: {self.output_features}')

    def save(self, path):
        np.save(os.path.join(path, 'w.npy'), self.w.numpy())
        np.save(os.path.join(path, 'b.npy'), self.b.numpy())
        
    def load(self, path):
        w = np.load(os.path.join(path, 'w.npy'))
        self.w.assign(w)
        del w
        b = np.load(os.path.join(path, 'b.npy'))
        self.b.assign(b)
        del b
    
    def reset(self):
        self.w.assign(tf.random.normal([self.input_features, self.output_features],
                                       dtype=tf.float32,
                                       mean=0.0,
                                       stddev=self.scaling))
        self.b.assign(tf.zeros([self.output_features], dtype=tf.float32))
    
    @tf.function
    def __call__(self, x):
        y = tf.tensordot(x, self.w, axes=1) + self.b
        return tf.nn.relu(y)
    
class ReluNormLayer(tf.Module):
    def __init__(self, input_features, output_features, scaling=None, name=None):
        super(ReluNormLayer, self).__init__(name=name)
        self.input_features = input_features
        self.output_features = output_features
        if scaling:
            self.scaling = tf.constant(scaling, dtype=tf.float32)
        else:
            self.scaling = tf.constant(np.sqrt(2 / self.input_features), dtype=tf.float32)
        self.w = tf.Variable(tf.random.normal([self.input_features, self.output_features],
                                              dtype=tf.float32,
                                              mean=0.0,
                                              stddev=self.scaling), name='w')
        self.gamma = tf.Variable(tf.ones([self.output_features], dtype=tf.float32), name='gamma')
        self.beta = tf.Variable(tf.zeros([self.output_features], dtype=tf.float32), name='beta')
        print(f'RELU Layer. Input Features: {self.input_features} Output Features: {self.output_features}')

    def save(self, path):
        np.save(os.path.join(path, 'w.npy'), self.w.numpy())
        np.save(os.path.join(path, 'gamma.npy'), self.gamma.numpy())
        np.save(os.path.join(path, 'beta.npy'), self.beta.numpy())
        
    def load(self, path):
        w = np.load(os.path.join(path, 'w.npy'))
        self.w.assign(w)
        del w
        gamma = np.load(os.path.join(path, 'gamma.npy'))
        self.gamma.assign(gamma)
        del gamma
        beta = np.load(os.path.join(path, 'beta.npy'))
        self.beta.assign(beta)
        del beta
    
    def reset(self):
        self.w.assign(tf.random.normal([self.input_features, self.output_features],
                                       dtype=tf.float32,
                                       mean=0.0,
                                       stddev=self.scaling))
        self.gamma.assign(tf.ones([self.output_features], dtype=tf.float32))
        self.beta.assign(tf.zeros([self.output_features], dtype=tf.float32))
    
    @tf.function
    def __call__(self, x):
        y = tf.tensordot(x, self.w, axes=1)
        z = tf.nn.batch_normalization(y, *tf.nn.moments(y, [0]), self.beta, self.gamma, 10**(-8))
        return tf.nn.relu(z)
    
    
class SeluLayer(tf.Module):
    def __init__(self, input_features, output_features, scaling=None, name=None):
        super(SeluLayer, self).__init__(name=name)
        self.input_features = input_features
        self.output_features = output_features
        if scaling:
            self.scaling = tf.constant(scaling, dtype=tf.float32)
        else:
            self.scaling = tf.constant(np.sqrt(2 / self.input_features), dtype=tf.float32)
        self.w = tf.Variable(tf.random.normal([self.input_features, self.output_features],
                                              dtype=tf.float32,
                                              mean=0.0,
                                              stddev=self.scaling), name='w')
        self.b = tf.Variable(tf.zeros([self.output_features], dtype=tf.float32), name='b')
        print(f'SELU Layer. Input Features: {self.input_features} Output Features: {self.output_features}')

    def save(self, path):
        np.save(os.path.join(path, 'w.npy'), self.w.numpy())
        np.save(os.path.join(path, 'b.npy'), self.b.numpy())
        
    def load(self, path):
        w = np.load(os.path.join(path, 'w.npy'))
        self.w.assign(w)
        del w
        b = np.load(os.path.join(path, 'b.npy'))
        self.b.assign(b)
        del b
        
    def reset(self):
        self.w.assign(tf.random.normal([self.input_features, self.output_features],
                                       dtype=tf.float32,
                                       mean=0.0,
                                       stddev=self.scaling))
        self.b.assign(tf.zeros([self.output_features], dtype=tf.float32))
    
    @tf.function
    def __call__(self, x):
        y = tf.tensordot(x, self.w, axes=1) + self.b
        return tf.nn.selu(y)

class SigmoidLayer(tf.Module):
    def __init__(self, input_features, output_features, scaling=None, name=None):
        super(SigmoidLayer, self).__init__(name=name)
        self.input_features = input_features
        self.output_features = output_features
        if scaling:
            self.scaling = tf.constant(scaling, dtype=tf.float32)
        else:
            self.scaling = tf.constant(np.sqrt(1 / self.input_features), dtype=tf.float32)
        self.w = tf.Variable(tf.random.normal([self.input_features, self.output_features],
                                              dtype=tf.float32,
                                              mean=0.0,
                                              stddev=self.scaling), name='w')
        self.b = tf.Variable(tf.zeros([self.output_features], dtype=tf.float32), name='b')
        print(f'Sigmoid Layer. Input Features: {self.input_features} Output Features: {self.output_features}')

    def save(self, path):
        np.save(os.path.join(path, 'w.npy'), self.w.numpy())
        np.save(os.path.join(path, 'b.npy'), self.b.numpy())
        
    def load(self, path):
        w = np.load(os.path.join(path, 'w.npy'))
        self.w.assign(w)
        del w
        b = np.load(os.path.join(path, 'b.npy'))
        self.b.assign(b)
        del b
        
    def reset(self):
        self.w.assign(tf.random.normal([self.input_features, self.output_features],
                                       dtype=tf.float32,
                                       mean=0.0,
                                       stddev=self.scaling))
        self.b.assign(tf.zeros([self.output_features], dtype=tf.float32))
    
    
    @tf.function
    def __call__(self, x):
        y = tf.tensordot(x, self.w, axes=1) + self.b
        return tf.nn.sigmoid(y)
    
    
class SigmoidNormLayer(tf.Module):
    def __init__(self, input_features, output_features, scaling=None, name=None):
        super(SigmoidNormLayer, self).__init__(name=name)
        self.input_features = input_features
        self.output_features = output_features
        if scaling:
            self.scaling = tf.constant(scaling, dtype=tf.float32)
        else:
            self.scaling = tf.constant(np.sqrt(2 / self.input_features), dtype=tf.float32)
        self.w = tf.Variable(tf.random.normal([self.input_features, self.output_features],
                                              dtype=tf.float32,
                                              mean=0.0,
                                              stddev=self.scaling), name='w')
        self.gamma = tf.Variable(tf.ones([self.output_features], dtype=tf.float32), name='gamma')
        self.beta = tf.Variable(tf.zeros([self.output_features], dtype=tf.float32), name='beta')
        print(f'RELU Layer. Input Features: {self.input_features} Output Features: {self.output_features}')

    def save(self, path):
        np.save(os.path.join(path, 'w.npy'), self.w.numpy())
        np.save(os.path.join(path, 'gamma.npy'), self.gamma.numpy())
        np.save(os.path.join(path, 'beta.npy'), self.beta.numpy())
        
    def load(self, path):
        w = np.load(os.path.join(path, 'w.npy'))
        self.w.assign(w)
        del w
        gamma = np.load(os.path.join(path, 'gamma.npy'))
        self.gamma.assign(gamma)
        del gamma
        beta = np.load(os.path.join(path, 'beta.npy'))
        self.beta.assign(beta)
        del beta
    
    def reset(self):
        self.w.assign(tf.random.normal([self.input_features, self.output_features],
                                       dtype=tf.float32,
                                       mean=0.0,
                                       stddev=self.scaling))
        self.gamma.assign(tf.ones([self.output_features], dtype=tf.float32))
        self.beta.assign(tf.zeros([self.output_features], dtype=tf.float32))
    
    @tf.function
    def __call__(self, x):
        y = tf.tensordot(x, self.w, axes=1)
        z = tf.nn.batch_normalization(y, *tf.nn.moments(y, [0]), self.beta, self.gamma, 10**(-8))
        return tf.nn.sigmoid(z)    
    
    
    
class DisperseLayer(tf.Module):
    def __init__(self, input_features, chunk_size, shift_percentage, name=None):
        super(DisperseLayer, self).__init__(name=name)
        self.scaling = tf.constant(1.0, dtype=tf.float32)
        self.input_features = input_features
        inverse_shift_ratio = int(round(1 / shift_percentage))
        self.num_chunks = (inverse_shift_ratio*(input_features - chunk_size) // chunk_size) + 1
        self.shift_size = chunk_size // inverse_shift_ratio
        self.chunk_size = chunk_size
        print(f'Disperse Layer. Input Features: {input_features} Chunk Size: {chunk_size} Num Chunks: {self.num_chunks}')
        
    def save(self, path):
        pass
    
    def load(self, path):
        pass
        
    def reset(self):
        pass
    
    @tf.function
    def __call__(self, x):
        output_array = tf.TensorArray(dtype=tf.float32, size=self.num_chunks)
        shape = x.get_shape()
        other_dims = shape.rank - 1
        for i in range(self.num_chunks):
            begin = [0 for i in range(other_dims)]+[i*self.shift_size]
            size = [-1 for i in range(other_dims)]+[self.chunk_size]
            output_array = output_array.write(i, tf.slice(x, begin, size))
        return output_array.stack()
    
class JoinLayer(tf.Module):
    def __init__(self, chunk_size, num_chunks, name=None):
        super(JoinLayer, self).__init__(name=name)
        self.scaling = tf.constant(1.0, dtype=tf.float32)
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.output_features = chunk_size * num_chunks
        print(f'Join Layer. Input Chunk Size: {chunk_size} Num Input Chunks: {num_chunks} Output Features: {self.output_features}')

    def save(self, path):
        pass
    
    def load(self, path):
        pass        
    def reset(self):
        pass
    
    @tf.function
    def __call__(self, x):
        players = tf.unstack(x, num=self.num_chunks, axis=0)
        return tf.concat(players, axis=-1)
        
# In a sparse layer, the input and outputs are each divided into an equal number
# of (probably overlapping) chunks. Every neuron in input chunk j is connected to
# every neuron in output chunk j. 
class SparseLayer(tf.Module):
    def __init__(self, input_features, input_chunk_size, input_chunk_step,
                 output_features, output_chunk_size, output_chunk_step, name=None):
        super(SparseLayer, self).__init__(name=name)
        self.scaling = tf.constant(np.sqrt(1 / input_chunk_size), dtype=tf.float32)
        if (input_features - input_chunk_size) % input_chunk_step != 0:
            raise ValueError('input_features must be of the form input_chunk_size + k*input_chunk_step for some k')
        if (output_features - output_chunk_size) % output_chunk_step != 0:
            raise ValueError('output_features must be of the form output_chunk_size + k*output_chunk_step for some k')
        if (input_chunk_step > input_chunk_size) or (output_chunk_step > output_chunk_size):
            raise ValueError('chunk step should be no bigger than chunk size')
        num_input_chunks = 1 + ((input_features - input_chunk_size) // input_chunk_step)
        num_output_chunks = 1 + ((output_features - output_chunk_size) // output_chunk_step)
        if num_input_chunks != num_output_chunks:
            raise ValueError('number of input chunks and output chunks does not match')
        self.num_chunks = num_input_chunks
        self.input_features = input_features
        self.input_chunk_size = input_chunk_size
        self.input_chunk_step = input_chunk_step
        self.output_features = output_features
        self.output_chunk_size = output_chunk_size
        self.output_chunk_step = output_chunk_step
        self.w = tf.Variable(tf.random.normal([num_chunks, input_chunk_size, output_chunk_size], dtype=tf.float32, stddev=self.scaling), name='w')
        self.b = tf.Variable(tf.zeros([num_chunks, output_chunk_size], dtype=tf.float32), name='b')
        print(f'Sparse Layer. Input Features: {input_features} Output Features: {self.output_features}')
        print(f'Input Chunk Size: {input_chunk_size} Input Chunk Step: {input_chunk_step}')
        print(f'Output Chunk Size: {output_chunk_size} Output Chunk Step: {output_chunk_step}')

        
    @tf.function
    def __call__(self, x):
        output_array = tf.TensorArray(dtype=tf.float32, size=self.num_chunks)
        
        input_shape = x.get_shape()
        other_dims = shape.rank - 1
        
        zeros_to_concat_shape = input_shape.as_list()
        zeros_to_concat_shape[-1] = output_features - output_chunk_size

        y = tf.zeros(output_shape, dtype=tf.float32)
        
        for i in range(self.num_chunks):
            begin = [0 for i in range(other_dims)]+[i*self.input_chunk_step]
            size = [-1 for i in range(other_dims)]+[chunk_size]
            chunk = tf.concat([tf.tensordot(tf.slice(x, begin, size), self.w[i], axes=1) + self.b[i],
                               tf.zeros(zeros_to_concat_shape, dtype=tf.float32)],
                              axis=-1)
            rolled_chunk = tf.roll(chunk, shift=i*self.output_chunk_step, axis=-1)
            output_array.write(i, rolled_chunk)
        pre_y = output_array.stack()
        y = tf.reduce_sum(pre_y, 0)
        return tf.nn.relu(y)
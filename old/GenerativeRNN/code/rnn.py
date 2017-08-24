import numpy as np
import time


class RNN:
    def __init__(self, vocab_size, loss_function, activation_function, hidden_size=100, sequence_length=25):
        self.hidden_size = hidden_size  # size of hidden layer of neurons
        self.sequence_length = sequence_length  # number of steps to unroll the RNN for
        self.vocab_size = vocab_size

        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((vocab_size, 1))  # output bias

        self.loss_function = loss_function
        self.activation_function = activation_function

        self.hprev = np.zeros((self.hidden_size, 1))

    def onehot(self, ix):
        x = np.zeros((self.vocab_size, 1))
        x[ix] = 1
        return x

    def forward(self, xs, keep_memory=False):
        hs, ys, ps = {}, {}, {}
        hs[-1] = np.copy(self.hprev)

        for t in range(len(xs)):
            hs[t] = self.activation_function(
                np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)  # hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by  # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars

        if keep_memory:
            self.hprev = hs
        else:
            self.hprev = hs[len(xs) - 1]
        return ps

    @staticmethod
    def cpu_temp():
        return int(open('/sys/class/thermal/thermal_zone0/temp', 'r').read().strip()) / 1000

    @staticmethod
    def activation_tanh(z, derivative=False):
        if derivative:
            return 1 - z * z
        return np.tanh(z)

    @staticmethod
    def activation_softsign(z, derivative=False):
        if derivative:
            return 1 / ((1 + abs(z)) * (1 + abs(z)))
        return z / (1 + abs(z))

    @staticmethod
    def activation_lrelu(leak_param):
        lrelu = lambda z, derivative=False: (leak_param * z if z < 0 else z) if not derivative \
            else (leak_param if z < 0 else 1.0)

        return np.vectorize(lrelu)

    @staticmethod
    def loss_softmax(ps, targets):
        # softmax (cross-entropy loss)
        return np.array([-np.log(ps[t][targets[t], 0]) for t in range(len(ps))]).sum()

    def backprop(self, xs, ps, targets):
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(self.hprev[0])

        for t in reversed(range(len(xs))):
            dy = np.copy(ps[t])
            # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, self.hprev[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
            dhraw = self.activation_function(self.hprev[t], derivative=True) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, self.hprev[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

        return dWxh, dWhh, dWhy, dbh, dby

    def adagrad(self, dWxh, dWhh, dWhy, dbh, dby, mWxh, mWhh, mWhy, mbh, mby, rho, lr):
        # perform parameter update with Adagrad
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -lr * dparam / np.sqrt(mem + rho)  # adagrad update

    def fit(self, data, ix_to_char, learning_rate=0.1, rho=1e-8, verbose=200, sample_size=200, max_iter=None,
            target_loss=None, throttle=lambda t: 1 if t < 85 else 2):
        # memory variables for Adagrad
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)
        n, p, l = 0, 0, 0
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.sequence_length  # loss at iteration 0

        while (target_loss is not None and target_loss > smooth_loss) or \
                (max_iter is not None and max_iter > n) or \
                (target_loss is None and max_iter is None):
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + self.sequence_length + 1 >= len(data) or n == 0:
                self.reset_memory()
                p = 0
                l += 1

            xs = [self.onehot(ch) for ch in data[p:p + self.sequence_length]]
            targets = data[p + 1:p + self.sequence_length + 1]

            ps = self.forward(xs, keep_memory=True)

            loss = self.loss_function(ps, targets)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            dWxh, dWhh, dWhy, dbh, dby = self.backprop(xs, ps, targets)
            self.adagrad(dWxh, dWhh, dWhy, dbh, dby, mWxh, mWhh, mWhy, mbh, mby, rho, learning_rate)

            self.hprev = self.hprev[-1]
            if verbose is not False and n % verbose == 0:
                temp = RNN.cpu_temp()
                time.sleep(throttle(temp))
                ixes = self.sample(0, data[p], sample_size)
                txt = ''.join(ix_to_char[ix] for ix in [data[p]] + ixes)
                print(
                    '========\n iter: %d, loss: %f, p: %d/%d (%dx %.1f%%), cpu temp: %d (throttle: %.1fs), seed: "%s"\n-----\n%s\n------\n'
                    % (
                        n, smooth_loss, p, len(data), l - 1, p / len(data) * 100, temp, throttle(temp),
                        ix_to_char[data[p]],
                        txt))

            p += self.sequence_length  # move data pointer
            n += 1  # iteration counter

    def reset_memory(self):
        self.set_hidden_state(np.zeros((self.hidden_size, 1)))

    def set_hidden_state(self, h):
        self.hprev = h

    def sample(self, h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        x = self.onehot(seed_ix)
        ixes = []
        for t in range(n):
            p = self.forward(np.array([x]))[0]

            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = self.onehot(ix)
            ixes.append(ix)

        return ixes


if __name__ == "__main__":
    # data I/O
    # should be simple plain text file
    import re

    if False:
        data = open('../FEII/full_reports/GENERAL-ELECTRIC_2013.htm', 'r', encoding='utf-8', errors='ignore').read()
    else:
        import os
        from email import parser as ep

        parser = ep.Parser()

        data = ''
        cnt = 0
        skip = 2000
        limit = 50
        maildir = '../../../../enron/data/original/'
        for root, dirs, files in os.walk(maildir):
            if cnt > limit + skip:
                break
            stripped = root[len(maildir):]
            print("entering %s containing %d files and %d dirs" % (stripped, len(files), len(dirs)))

            for file in files:
                with open(root + "/" + file, "r", encoding='utf-8', errors='ignore') as f:
                    cnt += 1
                    if cnt < skip:
                        continue
                    # data += parser.parsestr(f.read()).get_payload() + '\n'
                    data += re.sub(r'[a-z]', 'x', parser.parsestr(f.read()).get_payload() + '\n', flags=re.IGNORECASE)

    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    print("====================")
    print(data)

    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    data_i = [char_to_ix[ch] for ch in data]

    rnn = RNN(vocab_size, hidden_size=5, sequence_length=40,
              activation_function=RNN.activation_tanh,
              # activation_function=RNN.activation_softsign,
              # activation_function=RNN.activation_lrelu(0.01),
              loss_function=RNN.loss_softmax)
    rnn.fit(data_i, ix_to_char, learning_rate=0.1, max_iter=500000, verbose=500,
            throttle=lambda t: 0 if t < 85 else 2.5)

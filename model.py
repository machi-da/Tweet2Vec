import chainer
from chainer import links as L
from chainer import functions as F


class Tweet2vec(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout, class_size):
        super(Tweet2vec, self).__init__()
        with self.init_scope():
            self.encoder = Encoder(vocab_size, embed_size, hidden_size, dropout)
            self.proj = L.Linear(hidden_size, class_size)

    def __call__(self, sentences, gold_labels):
        hs, cs, enc_ys = self.encoder(None, None, sentences)
        proj_labels = self.proj(hs)
        loss = F.softmax_cross_entropy(proj_labels, gold_labels)
        return loss

    def predict(self, sentences, gold_labels):
        hs, cs, enc_ys = self.encoder(None, None, sentences)
        proj_labels = self.proj(hs)
        proj_labels = F.softmax(proj_labels).data
        proj_labels = self.xp.argmax(proj_labels, axis=1)
        return proj_labels


class Encoder(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        n_layers = 1
        super(Encoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, embed_size)
            self.Nlstm = L.NStepBiLSTM(n_layers, embed_size, hidden_size, dropout)
        self.hidden_size = hidden_size

    def __call__(self, hx, cx, xs):
        xs_embed = [self.embed(x) for x in xs]
        hy, cy, ys = self.Nlstm(hx, cx, xs_embed)

        hy = F.sum(hy, axis=0)
        cy = F.sum(cy, axis=0)
        ys = [F.sum(F.reshape(y, (-1, 2, self.hidden_size)), axis=1) for y in ys]

        return hy, cy, ys
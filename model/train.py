import tensorflow as tf
from model.mgccn import MGCCN
from utils.evaluate import cluster_acc, f_score, nmi, ari, err_rate


class Trainer():
    def __init__(self, args):
        self.args = args
        self.build_placeholders()
        self.model = MGCCN(self.args)
        self.loss, self.ft_loss, self.st_loss, self.cl_loss, \
            self.H, self.C, self.H2, self.C2, self.y_pred, self.z, self.z2 = \
            self.model(self.A, self.X,
                       self.R, self.S,
                       self.p, self.A2,
                       self.X2, self.R2, self.S2)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(dtype=tf.int64)
        self.R = tf.placeholder(dtype=tf.int64)
        self.A2 = tf.sparse_placeholder(dtype=tf.float32)
        self.X2 = tf.placeholder(dtype=tf.float32)
        self.S2 = tf.placeholder(dtype=tf.int64)
        self.R2 = tf.placeholder(dtype=tf.int64)
        self.p = tf.placeholder(dtype=tf.float32, shape=(None, self.args.cluster))

    def build_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.intra_op_parallelism_threads = 0
        # config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, X, S, R, A2, X2, S2, R2, y_true):
        for epoch in range(self.args.n_epochs):
            q = self.session.run(self.model.q, feed_dict={self.A: A, self.X: X, self.S: S, self.R: R,
                                                          self.A2: A2, self.X2: X2, self.S2: S2, self.R2: R2})
            p = self.model.target_distribution(q)
            loss, y_pred, _, z, z2 = self.session.run([self.loss, self.y_pred, self.train_op, self.z, self.z2],
                                                 feed_dict={self.A: A, self.X: X, self.S: S, self.R: R, self.p: p,
                                                            self.A2: A2, self.X2: X2, self.S2: S2, self.R2: R2})
            # missrate_x = err_rate(y_true, y_pred)
            print("Epoch--{}:\t\tloss: {:.8f}\t\tacc: {:.8f}\t\tnmi: {:.8f}\t\tf1: {:.8f}\t\tari: {:.8f}".
                        format(epoch, loss, cluster_acc(y_true, y_pred), nmi(y_true, y_pred),
                               f_score(y_true, y_pred), ari(y_true, y_pred)))

    def initialization(self, A, X, S, R, A2, X2, S2, R2):
        z, z2 = self.session.run([self.z, self.z2],
                                feed_dict={self.A: A, self.X: X, self.S: S, self.R: R,
                                           self.A2: A2, self.X2: X2, self.S2: S2, self.R2: R2})
        return z, z2

    def assign_cluster(self, A, X, S, R, A2, X2, S2, R2):
        embeddings, embeddings2 = self.initialization(A, X, S, R, A2, X2, S2, R2)
        assign_mu_op = self.model.get_assign_cluster_centers_op(
            (self.args.beta_1 * embeddings + self.args.beta_2 * embeddings2))
        _ = self.session.run(assign_mu_op)
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans


class MGCCN():
    def __init__(self, args):
        self.args = args
        self.lambda_1 = self.args.lambda_1
        self.lambda_2 = self.args.lambda_2
        self.lambda_3 = self.args.lambda_3
        self.n_layers1 = len(self.args.hidden_dims1) - 1
        self.n_layers2 = len(self.args.hidden_dims2) - 1
        # self.n_layers3 = len(self.args.hidden_dims2) - 1
        self.W, self.v = self.define_weights(self.args.hidden_dims1)
        self.C = {}
        self.W2, self.v2 = self.define_weights2(self.args.hidden_dims2)
        self.C2 = {}
        # self.W3, self.v3 = self.define_weights2(self.args.hidden_dims3)
        # self.C3 = {}
        self.mu = tf.Variable(tf.zeros(shape=(self.args.cluster, self.args.embedding)), name="mu")
        self.kmeans = KMeans(n_clusters=self.args.cluster, n_init=10, random_state=args.init)
        self.n_cluster = self.args.cluster
        self.input_batch_size = self.args.n_sample
        self.alpha = self.args.alpha

    def __call__(self, A, X, R, S, p, A2, X2, R2, S2):
        # Encoder1
        H = X
        for layer in range(self.n_layers1):
            H = self.__encoder(A, H, layer)
        # Final node representations
        self.H = H

        # Decoder1
        for layer in range(self.n_layers1 - 1, -1, -1):
            H = self.__decoder(H, layer)
        X_ = H

        self.Z = self.H

        # Encoder2
        H2 = X2
        for layer in range(self.n_layers2):
            H2 = self.__encoder2(A2, H2, layer)
        # Final node representations
        self.H2 = H2

        # Decoder2
        for layer in range(self.n_layers2 - 1, -1, -1):
            H2 = self.__decoder2(H2, layer)
        X_2 = H2

        self.Z2 = self.H2

        self.q = self._soft_assignment((self.args.beta_1 * self.Z + self.args.beta_2 * self.Z2), self.mu)
        self.p = p
        self.y_pred = tf.argmax(self.q, axis=1)

        # The reconstruction loss of node features
        self.ft_loss = tf.reduce_mean((X - X_) ** 2) + tf.reduce_mean((X2 - X_2) ** 2)
        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        self.S_emb2 = tf.nn.embedding_lookup(self.H2, S2)
        self.R_emb2 = tf.nn.embedding_lookup(self.H2, R2)
        self.st_loss1 = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))
        self.st_loss2 = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb2 * self.R_emb2, axis=-1)))
        self.st_loss = tf.reduce_sum(self.st_loss1) + tf.reduce_sum(self.st_loss2)
        # The loss of self-supervised clustering
        self.ssc_loss = self._self_supervised_clustering(self.p, self.q)
        self.cl_loss = self._constrastive_loss(self.Z, self.Z2, self.args.n_sample)
        # Total loss
        self.loss = self.ft_loss + self.lambda_1 * self.st_loss + self.lambda_2 * self.ssc_loss + self.lambda_3 * self.cl_loss

        return self.loss, self.ft_loss, self.st_loss, self.cl_loss, self.H, self.C, self.H2, self.C2, self.y_pred, self.Z, self.Z2

    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __encoder2(self, A, H, layer):
        H = tf.matmul(H, self.W2[layer])
        self.C2[layer] = self.graph_attention_layer2(A, H, self.v2[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C2[layer], H)

    def __decoder2(self, H, layer):
        H = tf.matmul(H, self.W2[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C2[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers1):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers1):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def define_weights2(self, hidden_dims):
        W2 = {}
        for i in range(self.n_layers2):
            W2[i] = tf.get_variable("W2%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att2 = {}
        for i in range(self.n_layers2):
            v2 = {}
            v2[0] = tf.get_variable("v2%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v2[1] = tf.get_variable("v2%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att2[i] = v2

        return W2, Ws_att2

    def graph_attention_layer(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

    def graph_attention_layer2(self, A, M, v, layer):
        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)
            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

    def get_assign_cluster_centers_op(self, features):
        print("initialize cluster centroids")
        kmeans = self.kmeans.fit(features)
        return tf.assign(self.mu, kmeans.cluster_centers_)

    def _soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.

        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)

        Return:
            q_i_j: (num_points, num_cluster)
        """

        def _pairwise_euclidean_distance(a, b):
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(self.input_batch_size, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))
            return res
        # print("embeddings.shape", embeddings.shape)
        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def _self_supervised_clustering(self, target, pred):
        return tf.reduce_mean((target - pred) ** 2)

    def _constrastive_loss(self, z_i, z_j, batch_size, temperature=1.0):
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        negative_mask = self.get_negative_mask(batch_size)
        zis = tf.nn.l2_normalize(z_i, axis=1)
        zjs = tf.nn.l2_normalize(z_j, axis=1)
        l_pos = self._dot_simililarity_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (batch_size, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0
        for positives in [zis, zjs]:
            l_neg = self._dot_simililarity_dim2(positives, negatives)

            labels = tf.zeros(batch_size, dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (batch_size, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1)
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * batch_size)
        return loss

    def _cosine_simililarity_dim1(self, x, y):
        cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
        v = cosine_sim_1d(x, y)
        return v

    def _cosine_simililarity_dim2(self, x, y):
        cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
        return v

    def _dot_simililarity_dim1(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (N, C, 1)
        # v shape: (N, 1, 1)
        v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
        return v

    def _dot_simililarity_dim2(self, x, y):
        v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def get_negative_mask(self, batch_size):
        # return a mask that removes the similarity score of equal/similar images.
        # this function ensures that only distinct pair of images get their similarity scores
        # passed as negative examples
        negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0
        return tf.constant(negative_mask)

    def __cosine_similarity(self, z, z2):
        z = tf.nn.l2_normalize(z, axis=1)
        z2 = tf.nn.l2_normalize(z2, axis=1)
        return tf.reduce_mean(tf.reduce_sum(-(z * z2), axis=1))






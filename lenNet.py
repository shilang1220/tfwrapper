import tensorflow as tf
from dataset import dataset,trainDataset,verifyDataset,testDataset

###########################################
# Realizition class of Lennet
###########################################
class lenNet(object):
    '''
    Main member attributions:
    X,Y:            训练/验证/测试数据集
    self.graph:          计算图所在绘画板
    self.total_loss:     损失函数（变量）
    self.optimizer：     优化器，默认为Adam优化器
    train_op:            优化训练操作，tf.run(train_op)
    '''
    def __init__(self,working_flag=0,dataset=None):
        #创建graphic
        self.graph = tf.graph()
        with self.graph.as_default():
            if working_flag == 0: # 训练
                if not dataset or not isinstance(dataset,trainDataset): return
                # set input data
                #dataset.get_next_batch()
                # build comput. graphic, export predicted y.
                self.build_arch()

                # define loss function, export cross_entroy,MSE etc.
                self.build_loss()

                # define optimizer
                self.optimizer = tf.train.AdamOptimizer()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_op = self.optimizer.minimize(self.total_loss,
                                                        global_step=self.global_step)  # var_list=t_vars)
                #define summary operator.
                self._summary()
                pass
            elif working_flag == 1: # 验证
                pass
            elif working_flag == 2: # 测试
                pass
            elif working_flag == 3: #预测
                pass
            else:
                tf.logging.info('Invalid parameter of working_flag: %d' %{working_flag})




    def build_network(self):
        pass

    def build_loss(self):
        pass

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    def predict(self):
        pass
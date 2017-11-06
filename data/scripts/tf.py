import tensorflow as tf
from images import get_all_image_data 

# Hidden layers.
l1_n = 2000
l2_n = 500
l3_n = 300

# Output layer.
n_classes = 2

training_steps = 50
learning_rate = 0.002

print(get_all_image_data())

def create_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([120000, l1_n])),
                      'biases':tf.Variable(tf.random_normal([l1_n]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([l1_n, l2_n])),
                      'biases':tf.Variable(tf.random_normal([l2_n]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([l2_n, l3_n])),
                      'biases':tf.Variable(tf.random_normal([l3_n]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([l3_n, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_model(data):
    prediction = create_model(data)
    error = tf.subtract(prediction, target)
    loss = tf.nn.l2_loss(error)
    tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_valiables())
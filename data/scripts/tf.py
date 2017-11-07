import tensorflow as tf
from images import get_all_image_data, get_test_image

input_n = 40000
# Hidden layers.
l1_n = 2000
l2_n = 500
l3_n = 300

# Output layer.
n_classes = 2

x = tf.placeholder('float')
y = tf.placeholder('float')

def create_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([input_n, l1_n])),
                      'biases': tf.Variable(tf.random_normal([l1_n]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([l1_n, l2_n])),
                      'biases': tf.Variable(tf.random_normal([l2_n]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([l2_n, l3_n])),
                      'biases': tf.Variable(tf.random_normal([l3_n]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([l3_n, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_model(x):
    train_data = get_all_image_data()
    train_images = [item[1] for item in train_data]
    train_labels = [item[0] for item in train_data]

    prediction = create_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            _, epoch_loss = sess.run([optimizer, cost], feed_dict = {x: train_images, y: train_labels})

            print('Epoch: ' + str(epoch) + ' Epoch loss: ' +  str(epoch_loss))


        # Check accuracy.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # Set [1, 0] for hot dog and [0, 1] for car.
        print('Accuracy:', accuracy.eval({x: [get_test_image()], y: [[1, 0]]}))

train_model(x)
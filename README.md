# model_application
# how to restore slim model
The TensorFlow save method saves three kinds of files because it stores the graph structure separately from the variable values. 
The .meta file describes the saved graph structure, so you need to import it before restoring the checkpoint (otherwise it doesn't know what variables the saved checkpoint values correspond to).

Even though there is no file named model.ckpt, you still refer to the saved checkpoint by that name when restoring it. From the saver.py source code: "Users only need to interact with the user-specified prefix... instead of any physical pathname."

ex:
# Recreate the EXACT SAME variables
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")

...

# Now load the checkpoint variable values
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "/tmp/model.ckpt")

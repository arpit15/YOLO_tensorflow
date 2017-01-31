import tensorflow as tf
from ipdb import set_trace


def load_model_from_ckpt(model, filename):
	reader = tf.train.NewCheckpointReader(filename)
	var_name = reader.get_variable_to_shape_map().keys()
	# print len(var_name)
	# set_trace()
	i = 0
	for var_num in range(len(var_name)/2):
		layer_name = model.layers[i].name
		while 'leakyrelu' in layer_name or 'pool' in layer_name or 'flatten' in layer_name:
			i +=1
			layer_name = model.layers[i].name

		# print layer_name,
		
		layer = model.layers[i]
		# print layer.get_weights()[0].shape, layer.get_weights()[1].shape
		# print "Variable_" + str(2*var_num + 2) + ":" + "Variable_" + str(2*var_num + 1)
		if var_num!= 0:
			curr_weight = [reader.get_tensor("Variable_" + str(2*var_num)), reader.get_tensor("Variable_" + str(2*var_num + 1))]
		else:
			curr_weight = [reader.get_tensor("Variable"), reader.get_tensor("Variable_" + str(2*var_num + 1))]
	
		model.layers[i].set_weights(curr_weight)
		i +=1


	## to print the name and shape of layer weights

	# for layer in model.layers:
	# 	if len(layer.get_weights()) != 0:
	# 		print layer.name,
	# 		print layer.get_weights()[0].shape, layer.get_weights()[1].shape


	# for name in var_name:
	# 	print name,
	# 	print reader.get_tensor(name).shape

	return model

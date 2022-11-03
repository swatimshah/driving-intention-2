import mne
import numpy
from sklearn.decomposition import PCA
from tensorflow.random import set_seed
from numpy.random import seed

# setting the seed
seed(1)
set_seed(1)

# Read complete data from the 'set' file in a dataframe and write it to 'csv' file

#file = mne.io.read_raw_eeglab("P01Data_Part1byEEGLab.set")
#complete_data = file.to_data_frame()
#complete_data.to_csv('complete_data.csv', index=False)

# Load the saved data again from the 'csv' file in numpy array  

loaded_complete_data = numpy.loadtxt('P01Data_Part1byEEGLab.csv', delimiter=',', skiprows=1)

# Read all the events from the data   

loaded_trigger_data = numpy.loadtxt('Trigger_p1Part1byIPG.csv', delimiter=',')

trigger_time = loaded_trigger_data[:, 2] 

print(trigger_time)

accumulated_trigger_time = numpy.empty(92)

accumulated_trigger_time[0] = trigger_time[0];

for i in range (1, 92):
	accumulated_trigger_time[i] = trigger_time[i] + accumulated_trigger_time[i-1];

print(accumulated_trigger_time)

# Load the Labels file which specify the "left", "right" and "straight" turns

labels_file = 'Labels.csv'
my_labels = numpy.loadtxt(labels_file, delimiter=',')
print(my_labels)
print(my_labels.shape)

# Combine events with number 4 and the labels

combined_events_and_labels = numpy.append(accumulated_trigger_time.reshape(92, 1), my_labels.reshape(3, 92).transpose(), axis=1)
print(combined_events_and_labels)

# Delete the unwanted columns from the events and labels data

combined_events_and_labels = numpy.delete(combined_events_and_labels, numpy.s_[1:3], axis=1)  
numpy.savetxt('combined_events_and_labels.csv', combined_events_and_labels, delimiter=',')	

# Pick an event from the 'combined' file. Check that event in the complete data. Extract 2 sec epoch from combined file for each event.

epochs = numpy.empty((0, 63))
input_data = numpy.empty((0, 63))
final_input_data = numpy.empty((0, 3969))
input_to_nn = numpy.empty((0, 3969))
complete_timestamps = loaded_complete_data[:, 0]
print(len(complete_timestamps))

for i in range (92):
	input_data_gathered = 0		
	my_event_time = combined_events_and_labels[i, 0]
	for j in range (len(complete_timestamps)):	
		if (my_event_time < complete_timestamps[j]):
			for k in range (1000):
				epochs = loaded_complete_data[j + k, 1:64]
				input_data = numpy.append(input_data, epochs.reshape(1, 63), axis=0)
				input_data_gathered = 1	
			print(input_data.shape)						

			my_pca = PCA(n_components=63, random_state=2)
			my_pca.fit(input_data)
			print(my_pca.components_.shape)
			input_to_nn = my_pca.components_.flatten().reshape(1, 3969)
			final_input_data = numpy.append(final_input_data, input_to_nn, axis=0)

			if (input_data_gathered == 1):
				input_data = numpy.empty((0, 63))
				break


final_input_data_with_labels = numpy.append(final_input_data, combined_events_and_labels[:, 1].reshape(92, 1), axis=1)
numpy.savetxt('final_input_data_with_labels.csv', final_input_data_with_labels, delimiter=',')	


import glob
import numpy as np
import csv
import sys
import math

def euclidean_dist(q1, q2):
    out = np.subtract(q1, q2)
    out = np.square(out)
    result_sum = np.sum(out)
    return math.sqrt(result_sum)


def array_sum(q1, q2):
	"""Return euclidean distance between two arrays q1 and q2"""
	if len(q1[0]) < len(q2[0]): # q2 is longer so pad q1
		resultOfPadding = np.zeros(q2[0].shape) 
		resultOfPadding[:q1[0].shape[0]] = q1[0]
		results = euclidean_dist(resultOfPadding, q2[0])
	elif len(q1[0]) > len(q2[0]): #q1 is longer so pad q2
		resultOfPadding = np.zeros(q1[0].shape)
		resultOfPadding[:q2[0].shape[0]] = q2[0]
		results = euclidean_dist(q1[0], resultOfPadding)
	else: 
		results = euclidean_dist(q1[0], q2[0])
	return results

def matrix_sum(m1, m2):
    """Return euclidean distance between two matrix m1 and m2"""
    m1_width, m1_height = m1.shape
    m2_width, m2_height = m2.shape
    distance_sum = 0
    print m1.shape
    print m2.shape
    if m1.shape == m2.shape:
        for i in range(0, m1_width):
            results = euclidean_dist(m1[i],m2[i])
            distance_sum = distance_sum + results
    elif m1_width == m2_width:
    	if m1_height < m2_height: # Need to pad m1
    		resultOfPadding = np.zeros(m2.shape)
    		resultOfPadding[:m1.shape[0], :m1.shape[1]] = m1
    		distance_sum = euclidean_dist(resultOfPadding, m2)

    	else: # m2_height < m1_height: # need to pad m2
    		resultOfPadding = np.zeros(m1.shape)
    		resultOfPadding[:m2.shape[0], :m2.shape[1]] = m2
    		distance_sum = euclidean_dist(m1, resultOfPadding)
    else:
        return sys.float_info.max
	return distance_sum

def search(self, query, limit):
        scores = {}
        for x in self.img_hist.iteritems():
            img_id = x[0]
            db_histogram = x[1]
            distance = euclidean_dist(query, db_histogram)
            scores[img_id] = distance

        heap = []
        for doc in scores:
            scores[doc] = scores[doc]
            heapq.heappush(heap, (scores[doc], doc))


        largest = heapq.nsmallest(limit, heap) # Filter to Top K results based on score
        return largest

def build_index(folder):
    """folder is like /feature/acoustic/feature_energy/ """
    data = []
    labels = []
    soundHist = []
    results = []
    for filename in glob.iglob(folder + '*.csv'):
        # print(filename)
        data = []
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            counter = 0
            for row in reader:
                if counter ==0:
                    labels.append(row[1])

                if counter !=0:
                    string = row[0].split(",")
                    for info in string:
                        soundHist.append(float(info))
                    data.append(np.asarray(soundHist))
                    soundHist=[]
                counter+=1

        results.append(np.asarray(data))
    return labels, results

if __name__ == '__main__':
    build_index()

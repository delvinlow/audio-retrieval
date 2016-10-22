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

    # sumVal = 0
    # for i in range(0, len(q1)):
    #     sumVal += np.power((q1[i] -q2[i]),2)
    # results = np.sqrt(sumVal)
    # return results

def array_sum(q1, q2):
    if len(q1[0]) == len(q2[0]):
        results = euclidean_dist(q1[0], q2[0])
    else:
        return sys.float_info.max
    return results

def matrix_sum(m1, m2):
    m1_width,m1_height = m1.shape
    m2_width,m2_height = m2.shape
    distance_sum = 0
    if m1.shape == m2.shape:
        for i in range(0, m1_width):
            results = euclidean_dist(m1[i],m2[i])
            distance_sum = distance_sum + results
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

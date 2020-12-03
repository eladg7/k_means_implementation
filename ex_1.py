import math
import sys

import scipy
import numpy as np
import scipy.io.wavfile


def find_closest_centroid(point, centroids):
    closest = math.inf
    closest_index = -1

    for index, centroid in enumerate(centroids):
        # calculate distance
        dist = np.linalg.norm(point - centroid)

        if dist < closest:
            closest = dist
            closest_index = index

    return closest_index


def update_centroid(clusters, centroids):
    #   updated means we've got to convergence
    updated = False
    for index, centroid in enumerate(centroids):
        cluster_len = len(clusters[index])
        if cluster_len > 0:
            new_point = np.sum(clusters[index], axis=0) / cluster_len
            rounded_values = round(new_point[0]), round(new_point[1])
            # update centroid only if it had changed
            if rounded_values[0] != centroid[0] or rounded_values[1] != centroid[1]:
                centroid[0], centroid[1] = rounded_values
                updated = True

    return updated


def k_means(points, k, centroids, output_list, epochs):
    #  create the clusters
    clusters = [[] for _ in range(k)]
    for epoch in range(epochs):
        clusters = [[] for _ in range(k)]
        #   calculate for each point what centroid should it be in
        for point in points:
            closest_centroid = find_closest_centroid(point, centroids)
            clusters[closest_centroid].append(point)

        updated = update_centroid(clusters, centroids)
        output_list.append(f"[iter {epoch}]:{','.join([str(i) for i in centroids])}")
        if not updated:
            #   we've got to convergence
            break
    return clusters, centroids


def print_file(output_list):
    with open('output.txt', 'w') as f:
        f.write("\n".join(output_list))


def main():
    try:
        sample, centroids = sys.argv[1], sys.argv[2]
    except IndexError:
        print('not enough arguments were supplied!')
        exit()
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)
    output_list = list()
    k_means(x, len(centroids), centroids, output_list, 30)
    print_file(output_list)


if __name__ == '__main__':
    main()

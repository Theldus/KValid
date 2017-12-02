/*
 * Copyright (C) 2017  Davidson Francis <davidsondfgl@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

/*
 *    ClusterEvaluator.java
 *    Written by Davidson Francis
 */

package weka.clusterers.kvalid;

import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.clusterers.AbstractClusterer;

/**
 * Generic interface for cluster evaluation.
 *
 * @author Davidson Francis (davidson.francis@sga.pucminas.br)
 * @version $Revision: 0001 $
 */
public interface ClusterEvaluator {

	/**
	 * Evaluates the clusterer after buildClusterer.
	 *
	 * @param clusterer        given clusterer.
	 * @param instances        dataset.
	 * @param distanceFunction distance function.
	 */
	void evaluate(AbstractClusterer clusterer, Instances centroids,
		Instances instances, DistanceFunction distanceFunction) throws Exception;
}

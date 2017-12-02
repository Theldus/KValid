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
 *    KValid.java
 *    Written by Davidson Francis
 */

package weka.clusterers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.Locale;

import weka.classifiers.rules.DecisionTableHashKey;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import weka.clusterers.kvalid.SilhouetteIndex;
import weka.clusterers.kvalid.GraphPlotter;

/**
 * <!-- globalinfo-start --> KValid: SimpleKMeans with validation.
 * For more information, see<br/>
 * <br/>
 * Davidson Francis (2017). <a href="http://github.com/Theldus/KValid">
 * https://github.com/Theldus/KValid/</a>
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <!-- options-end -->
 * 
 * @author Davidson Francis (davidson.francis@sga.pucminas.br)
 * @version $Revision: 0001 $
 */
public class KValid extends RandomizableClusterer implements
  NumberOfClustersRequestable, WeightedInstancesHandler {

	/** for serialization */
	static final long serialVersionUID = -217733168493644444L;

	/** SimpleKMeans */
	protected SimpleKMeans m_skmeans;

	/** Number of clusters. */
	protected int m_numClusters = 3;

	/** Distance function. */
	protected DistanceFunction m_distanceFunction = new EuclideanDistance();

	/** Maximum interations. */
	protected int m_maxInteration = 500;

	/** Initialization attributes. */
	public static final int RANDOM           = 0;
	public static final int KMEANS_PLUS_PLUS = 1;
	public static final int CANOPY           = 2;
	public static final int FARTHEST_FIRST   = 3;
	
	/** Validation attributes. */
	public static final int SILHOUETTE_INDEX = 0;
	public static final int DAVIES_BOULDIN   = 1;

    /** Validation method to use. */
    protected int m_validationMethod = SILHOUETTE_INDEX;

	/** Validation attributes. */
	public static final Tag[] VALIDATION_SELECTION = {
		new Tag(SILHOUETTE_INDEX, "Silhouette Index"),
		new Tag(DAVIES_BOULDIN, "Davies-Bouldin Index") };

	/** The initialization method to use */
	protected int m_initializationMethod = weka.clusterers.SimpleKMeans.RANDOM;

	/** My instances. */
	protected Instances m_instances = null;

	/** Minimum k. */
	protected int m_minimumK = 3;

	/** Maximum k. */
	protected int m_maximumK = 10;

	/** Cascade. */
	protected boolean m_cascade = false;

	/** SilhouetteIndex. */
	protected ArrayList<SilhouetteIndex> m_silhouetteIdx;

	/** Best K. */
	protected int m_bestK = 0;

	/** Show graph?. */
	protected boolean m_showGraph = false;

	/** Default constructor. */
	public KValid() {
		super();
		m_SeedDefault = 10;
		setSeed(m_SeedDefault);
	}

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {

		return "KValid: SimpleKMeans with cluster validation. More "
				+ "information, visit https://github.com/Theldus/KValid";
	}

	/**
	 * Gets the default capabilities of the clusterer.
	 * 
	 * @return Returns the capabilities of this clusterer
	 */
	@Override
	public Capabilities getCapabilities() {

		Capabilities result = super.getCapabilities();
		result.disableAll();
		result.enable(Capability.NO_CLASS);

		/* Atributes. */
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		return result;
	}

	/**
	 * Generates a clusterer.
	 * 
	 * @param data set of instances serving as training data
	 * @throws Exception if the clusterer has not been generated successfully
	 */
	@Override
	public void buildClusterer(Instances data) throws Exception {
		int start   = m_numClusters;
		int end     = m_numClusters;
		m_instances = data;

		if (m_cascade == true) {
			
			if (m_minimumK >= m_maximumK || m_minimumK < 2 || m_maximumK < 3)
				throw new Exception
					("Wrong minimum/maximum values, minimum should be >= 2 and maximum >= 3");
			
			start = m_minimumK;
			end   = m_maximumK;
		}

		m_silhouetteIdx = new ArrayList<SilhouetteIndex>();

		/* Cascade k-Means. */
		for (int i = start; i <= end; i++) {

			m_skmeans = new SimpleKMeans();

			/* Setup the configs. */
			m_skmeans.setInitializationMethod(new SelectedTag(m_initializationMethod,
				weka.clusterers.SimpleKMeans.TAGS_SELECTION));

			/* Set seed. */
			m_skmeans.setSeed(m_SeedDefault);

			/* Num clusters. */
			m_skmeans.setNumClusters(i);
			
			/* Distance function. */
			m_skmeans.setDistanceFunction(m_distanceFunction);
			
			/* Max iterations. */
			m_skmeans.setMaxIterations(m_maxInteration);

			/* Build clusterer. */
			m_skmeans.buildClusterer(data);

			/* Gets the validation, Silhouette or something else. */
			if (m_validationMethod == SILHOUETTE_INDEX) {

				m_silhouetteIdx.add(new SilhouetteIndex());
				
				m_silhouetteIdx.get(i - start).evaluate(m_skmeans, m_skmeans.getClusterCentroids(),
					m_instances, m_distanceFunction);
			}
		}

		/* Gets the 'best' K if cascade enable. */
		if (m_cascade == true) {
			double si = 0;
			
			if (m_validationMethod == SILHOUETTE_INDEX) {
				for (int i = 0; i < m_silhouetteIdx.size(); i++) {
					if (m_silhouetteIdx.get(i).getGlobalSilhouette() > si) {
						si  = m_silhouetteIdx.get(i).getGlobalSilhouette();
						m_bestK = i;
					}
				}
			}

			/* Repeats the clustering for the best K. */
			m_bestK += start;
			m_skmeans = new SimpleKMeans();
			
			/* Initialize and run k-Means again. */
			m_skmeans.setInitializationMethod(new SelectedTag(m_initializationMethod,
				weka.clusterers.SimpleKMeans.TAGS_SELECTION));
			
			m_skmeans.setSeed(m_SeedDefault);
			m_skmeans.setNumClusters(m_bestK);
			m_skmeans.setDistanceFunction(m_distanceFunction);
			m_skmeans.setMaxIterations(m_maxInteration);
			m_skmeans.buildClusterer(data);
			setNumClusters(m_bestK);
		}
	}

	/**
	 * Classifies a given instance.
	 * 
	 * @param instance the instance to be assigned to a cluster
	 * @return the number of the assigned cluster if the class is
	 *         enumerated, otherwise the predicted value.
	 * @throws Exception if the instance could not be classified successfully.
	 */
	@Override
	public int clusterInstance(Instance instance) throws Exception {
		if (m_skmeans == null)
			throw new Exception("The clusterer was not build yet!");
		else
			return m_skmeans.clusterInstance(instance);
	}

	/**
	 * Gets the tip text for this property.
	 *
	 * @return Property tip text.
	 */
	public String initializationMethodTipText() {
		return "The initialization method to use.";
	}

	/**
	 * Gets the initialization method to use
	 * 
	 * @return method the initialization method to use
	 */
	public SelectedTag getInitializationMethod() {
		return new SelectedTag(m_initializationMethod,
			weka.clusterers.SimpleKMeans.TAGS_SELECTION);
	}

	/**
	 * Sets the initialization method to use
	 * 
	 * @param method the initialization method to use
	 */
	public void setInitializationMethod(SelectedTag method) {
		if (method.getTags() == weka.clusterers.SimpleKMeans.TAGS_SELECTION) {
			m_initializationMethod = method.getSelectedTag().getID();
		}
	}

	/**
	 * Gets the tip text for this property.
	 *
	 * @return Property tip text.
	 */
	public String validationMethodTipText() {
		return "Which validation method: Silhouette Index or Davies-Bouldin Index";
	}

	/**
	 * Get sthe validation method to use
	 * 
	 * @return method the validation method to use
	 */
	public SelectedTag getValidationMethod() {
		return new SelectedTag(m_validationMethod, VALIDATION_SELECTION);
	}

	/**
	 * Sets the validation method to use
	 * 
	 * @param method the validation method to use
	 */
	public void setValidationMethod(SelectedTag vmethod) {
		if (vmethod.getTags() == VALIDATION_SELECTION) {
			m_validationMethod = vmethod.getSelectedTag().getID();
		}
	}

	/**
	 * Returns the number of clusters.
	 * 
	 * @return Returns the number of clusters generated.
	 * @throws Exception if number of clusters could not be returned successfully.
	 */
	@Override
	public int numberOfClusters() throws Exception {
		return m_numClusters;
	}

	/**
	 * Gets the tip text for this property.
	 *
	 * @return Property tip text.
	 */
	public String numClustersTipText() {
		return "Set the number of clusters";
	}

	/**
	 * Gets the number of clusters.
	 *
	 * @return Returns the number of clusters
	 */
	public int getNumClusters() {
		return m_numClusters;
	}

	/**
     * Sets the number of clusters to build the cluster.
	 * 
	 * @param n number of clusters
	 * @throws Exception if number of clusters is negative
	 */
	@Override
	public void setNumClusters(int n) throws Exception {
		if (n <= 0) {
			throw new Exception("Number of clusters must be > 0");
		}
		m_numClusters = n;
	}

	/**
	 * Gets the tip text for this property.
	 *
	 * @return Property tip text.
	 */
	public String maxIterationsTipText() {
		return "Set the maximum number of iterations";
	}

	/**
	 * Gets the maximum number of iterations.
	 *
	 * @return Maximum number of iterations.
	 */
	public int getMaxIterations() {
		return m_maxInteration;
	}

	/**
	 * Sets the maximum number of iterations.
	 *
	 * @param m maximum iterations.
	 * @throws Exception if the maximum number of iterations if < 1.
	 */
	public void setMaxIterations(int m) throws Exception {
		if (m < 1)
			throw new Exception("Number of iterations should be > 1");

		m_maxInteration = m;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String distanceFunctionTipText() {
		return "The distance function to use for comparison";
	}

	/**
	 * Returns the distance function currently in use.
	 * 
	 * @return the distance function
	 */
	public DistanceFunction getDistanceFunction() {
		return m_distanceFunction;
	}

	/**
	 * Sets the distance function to use for instance comparison.
	 * 
	 * @param df the new distance function to use
	 * @throws Exception if instances cannot be processed
	 */
	public void setDistanceFunction(DistanceFunction df) throws Exception {
		if (!(df instanceof EuclideanDistance) && !(df instanceof ManhattanDistance)) {
			throw new Exception(
				"KValid currently only supports the Euclidean and Manhattan distances."); 
		}
		m_distanceFunction = df;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String minimumKTipText() {
		return "The minimum K value for test when cascade option is enabled";
	}

	/**
	 * Returns the minimum K value when cascade k-means enabled.
	 *
	 * @return the minimum k value for test.
	 */
	public int getMinimumK() {
		return m_minimumK;
	}

	/**
	 * Sets the minimum K value. 
	 *
	 * @param minimumK minimum K value.
	 */
	public void setMinimumK(int minimumK) throws Exception { 
		m_minimumK = minimumK;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String maximumKTipText() {
		return "The maximum K value for test when cascade option is enabled";
	}

	/**
	 * Returns the maximum K value when cascade k-means enabled.
	 *
	 * @return the maximum k value for test.
	 */
	public int getMaximumK() {
		return m_maximumK;
	}

	/**
	 * Sets the maximum K value. 
	 *
	 * @param maximumK minimum K value.
	 */
	public void setMaximumK(int maximumK) throws Exception {
		m_maximumK = maximumK;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String cascadeTipText() {
		return "Cascade test: Tries to find the best K given a minimum/maximum value!";
	}

	/**
	 * Returns the cascade option selected.
	 *
	 * @return true if the cascade option is enabled, false otherwise.
	 */
	public boolean getCascade() {
		return m_cascade;
	}

	/**
	 * Enables/Disables the cascade option, i.e: attemp to find
	 * the best K.
	 *
	 * @param cascade Enables/Disables the cascade mode.
	 */
	public void setCascade(boolean cascade) {
		m_cascade = cascade;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String setShowGraphTipText() {
		return "Show graph: shows the graph representing the results when in cascade mode!";
	}

	/**
	 * Returns the graph option selected.
	 *
	 * @return true if the graph mode is enabled, false otherwise.
	 */
	public boolean getShowGraph() {
		return m_showGraph;
	}

	/**
	 * Enables/Disables the graph option.
	 *
	 * @param showGraph Enables/Disables the graph drawing.
	 */
	public void setShowGraph(boolean showGraph) {
		m_showGraph = showGraph;
	}

	/**
	 * Gets the current settings of KValid.
	 * 
	 * @return An array of strings suitable for passing to setOptions()
	 */
	@Override
	public String[] getOptions() {

		Vector<String> result = new Vector<String>();

		result.add("-init");
		result.add("" + getInitializationMethod().getSelectedTag().getID());

		result.add("-N");
		result.add("" + getNumClusters());

		result.add("-A");
		result.add((m_distanceFunction.getClass().getName() + " " + Utils
			.joinOptions(m_distanceFunction.getOptions())).trim());

		result.add("-I");
		result.add("" + getMaxIterations());

		result.add("-validation");
		result.add("" + getValidationMethod().getSelectedTag().getID());

		if (m_cascade) {

			result.add("-cascade");

			result.add("-minK");
			result.add("" + getMinimumK());

			result.add("-maxK");
			result.add("" + getMaximumK());
		}

		if (m_showGraph)
			result.add("-show-graph");

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Parse the options.
	 *
	 * @param options The list of options.
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		String temp;

		/* Initialization method. */
		temp = Utils.getOption("init", options);
		if (temp.length() > 0)
			setInitializationMethod(new SelectedTag(Integer.parseInt(temp),
				weka.clusterers.SimpleKMeans.TAGS_SELECTION));

		/* Num cluster. */
		temp = Utils.getOption("N", options);
		if (temp.length() > 0)
			setNumClusters(Integer.parseInt(temp));

		/* Distance function. */
		String distFunctionClass = Utils.getOption('A', options);
		if (distFunctionClass.length() != 0) {

			String distFunctionClassSpec[] = Utils.splitOptions(distFunctionClass);
			if (distFunctionClassSpec.length == 0)
				throw new Exception("Invalid DistanceFunction specification string.");
			
			String className = distFunctionClassSpec[0];
			distFunctionClassSpec[0] = "";

			setDistanceFunction((DistanceFunction) Utils.forName(
				DistanceFunction.class, className, distFunctionClassSpec));
		}
		else
			setDistanceFunction(new EuclideanDistance());

		/* Max iterations. */
		temp = Utils.getOption("I", options);
		if (temp.length() > 0)
			setMaxIterations(Integer.parseInt(temp));

		/* Validation method: Silhouette... */
		temp = Utils.getOption("validation", options);
		if (temp.length() > 0)
			setValidationMethod(new SelectedTag(Integer.parseInt(temp),
				VALIDATION_SELECTION));

		/* Tries to find the best K or not. */
		if ( (m_cascade = Utils.getFlag("cascade", options)) == true ) {
			
			temp = Utils.getOption("minK", options);
			if (temp.length() > 0)
				setMinimumK(Integer.parseInt(temp));

			temp = Utils.getOption("maxK", options);
			if (temp.length() > 0)
				setMaximumK(Integer.parseInt(temp));
		}

		/* Show graph option. */
		m_showGraph = Utils.getFlag("show-graph", options);

		super.setOptions(options);
		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Returns a string describing the results.
	 *
	 * @return a string describing the clusterer.
	 */
	@Override
	public String toString() {
		if (m_skmeans == null) 
			return "I don't have any clusterer yet!";

		StringBuffer description = new StringBuffer("KValid\n");
		description.append("======\n\n");

		description.append("=== Clustering validation, using: " +
			((m_validationMethod == SILHOUETTE_INDEX) ? "Silhouette Index"
			: "Davies-Bouldin Index") + " ===");

		if (m_validationMethod == SILHOUETTE_INDEX) {
			int start   = m_numClusters;
			int end     = m_numClusters;

			if (m_cascade == true) {
				start = m_minimumK;
				end   = m_maximumK;
			}

			description.append("\n");

			for (int i = start; i <= end; i++) {
				description.append("\nFor k = " + i + "\n");
				description.append( m_silhouetteIdx.get(i - start).toString() + "\n");
			}

			if (m_cascade == true) {
				description.append("\n~~ Best K: " + m_bestK + " ~~");
				description.append(
					"\nPlease manually check your dataset to figure out if this is really the best K");
			
				/* Show the graph. */
				ArrayList<Double> dataSet = new ArrayList<Double>();
				for (int i = 0; i < m_silhouetteIdx.size(); i++)
					dataSet.add( m_silhouetteIdx.get(i).getGlobalSilhouette() );

				/* Show the graph if needed. */
				if (m_showGraph == true) {
					GraphPlotter gp = new GraphPlotter("KValid - Silhouette Index");
					gp.plot(dataSet, m_minimumK, "Silhouette analysis for KMeans",
						"for k ranging between " + m_minimumK + " and " + m_maximumK,
						"k - value", "Silhouette Index");
				}
			}
		}
		
		description.append("\n\n");
		description.append( m_skmeans.toString() );

		return description.toString();
	}

	/**
	 * Main method for executing this class.
     * 
     * @param args use -h to list all parameters
     */
	public static void main(String[] args) {
		runClusterer(new KValid(), args);
	}
}

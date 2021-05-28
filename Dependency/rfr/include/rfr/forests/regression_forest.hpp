#ifndef RFR_REGRESSION_FOREST_HPP
#define RFR_REGRESSION_FOREST_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <tuple>
#include <utility>
#include <cmath>
#include <numeric>
#include <tuple>
#include <random>
#include <algorithm>
#include <functional>
#include <memory>
#include <assert.h>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>

#include <iostream>
#include <sstream>



#include "rfr/trees/tree_options.hpp"
#include "rfr/forests/forest_options.hpp"
#include "rfr/util.hpp"

namespace rfr{ namespace forests{

typedef cereal::PortableBinaryInputArchive binary_iarch_t;
typedef cereal::PortableBinaryOutputArchive binary_oarch_t;

typedef cereal::JSONInputArchive ascii_iarch_t;
typedef cereal::JSONOutputArchive ascii_oarch_t;



template <typename tree_type, typename num_t = float, typename response_t = float, typename index_t = unsigned int,  typename rng_type=std::default_random_engine>
class regression_forest{
  protected:
	std::vector<tree_type> the_trees;
	index_t num_features;
	index_t num_data_points;  // number of data points to train the random forest

	std::vector<std::vector<num_t> > bootstrap_sample_weights;
	
	num_t oob_error = NAN;
	
	// the forest needs to remember the data types on which it was trained
	std::vector<index_t> types;
	std::vector< std::array<num_t,2> > bounds;
	

  public:

	forest_options<num_t, response_t, index_t> options;


  	/** \brief serialize function for saving forests with cerial*/
  	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( options, the_trees, num_features, bootstrap_sample_weights, oob_error, types, bounds);
	}

	regression_forest(): options()	{}
	
	regression_forest(forest_options<num_t, response_t, index_t> opts): options(opts){}

	virtual ~regression_forest()	{};

	/**\brief growing the random forest for a given data set
	 * 
	 * \param data a filled data container
	 * \param rng the random number generator to be used
	 */
	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, rng_type &rng){

		if (options.num_trees <= 0)
			throw std::runtime_error("The number of trees has to be positive!");

		if ((!options.do_bootstrapping) && (data.num_data_points() < options.num_data_points_per_tree))
			throw std::runtime_error("You cannot use more data points per tree than actual data point present without bootstrapping!");


		types.resize(data.num_features());
		bounds.resize(data.num_features());
		for (auto i=0u; i<data.num_features(); ++i){
			types[i] = data.get_type_of_feature(i);
			auto p = data.get_bounds_of_feature(i);
			bounds[i][0] = p.first;
			bounds[i][1] = p.second;
		}

		the_trees.resize(options.num_trees);

        num_data_points = data.num_data_points();
		std::vector<index_t> data_indices(num_data_points);
		std::iota(data_indices.begin(), data_indices.end(), 0);

		num_features = data.num_features();
		
		// catch some stupid things that will make the forest crash when fitting
		if (options.num_data_points_per_tree == 0)
			throw std::runtime_error("The number of data points per tree is set to zero!");
		
		if (options.tree_opts.max_features == 0)
			throw std::runtime_error("The number of features used for a split is set to zero!");
		
		bootstrap_sample_weights.clear();

		for (auto &tree : the_trees){
            std::vector<num_t> bssf (data.num_data_points(), 0); // BootStrap Sample Frequencies
			// prepare the data(sub)set
			if (options.do_bootstrapping){
                std::uniform_int_distribution<index_t> dist (0,data.num_data_points()-1);
                for (auto i=0u; i < options.num_data_points_per_tree; ++i){
					bssf[dist(rng)]+=1;
				}
			}
			else{
				std::shuffle(data_indices.begin(), data_indices.end(), rng);
                for (auto i=0u; i < options.num_data_points_per_tree; ++i)
                    bssf[data_indices[i]] += 1;
			}
			
			tree.fit(data, options.tree_opts, bssf, rng);
			
			// record sample counts for later use
			if (options.compute_oob_error)
				bootstrap_sample_weights.push_back(bssf);
		}
		
		oob_error = NAN;
		
		if (options.compute_oob_error){
			
			rfr::util::running_statistics<num_t> oob_error_stat;
			
			for (auto i=0u; i < data.num_data_points(); i++){

				rfr::util::running_statistics<num_t> prediction_stat;

				for (auto j=0u; j<the_trees.size(); j++){
					// only consider data points that were not part of that bootstrap sample
					if (bootstrap_sample_weights[j][i] == 0)
						prediction_stat.push(the_trees[j].predict( data.retrieve_data_point(i)));
				}
				
				// compute squared error of prediction

				if (prediction_stat.number_of_points() > 0u){
					oob_error_stat.push(std::pow(prediction_stat.mean() - data.response(i), (num_t) 2));
				}
			}
			oob_error = std::sqrt(oob_error_stat.mean());
		}
	}


	/* \brief combines the prediction of all trees in the forest
	 *
	 * Every random tree makes an individual prediction which are averaged for the forest's prediction.
	 *
	 * \param feature_vector a valid vector containing the features
	 * \return response_t the predicted value
	 */
    response_t predict( const std::vector<num_t> &feature_vector) const{

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> mean_stats;
		for (auto &tree: the_trees)
			mean_stats.push(tree.predict(feature_vector));
		return(mean_stats.mean());
	}
    
    
   /* \brief makes a prediction for the mean and a variance estimation
    * 
    * Every tree returns the mean and the variance of the leaf the feature vector falls into.
    * These are combined to the forests mean prediction (mean of the means) and a variance estimate
    * (mean of the variance + variance of the means).
    * 
    * Use weighted_data = false if the weights assigned to each data point were frequencies, not importance weights.
    * Use this if you haven't assigned any weigths, too.
    * 
	* \param feature_vector a valid feature vector
	* \param weighted_data whether the data had importance weights
	* \return std::pair<response_t, num_t> mean and variance prediction
    */
    std::pair<num_t, num_t> predict_mean_var( const std::vector<num_t> &feature_vector, bool weighted_data = false){

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> mean_stats, var_stats;
		for (auto &tree: the_trees){
			auto stat = tree.leaf_statistic(feature_vector);
			mean_stats.push(stat.mean());
			if (stat.number_of_points() > 1){
				if (weighted_data) var_stats.push(stat.variance_unbiased_importance());
				else var_stats.push(stat.variance_unbiased_frequency());
			} else{
				var_stats.push(0);
			}
		}
	    num_t var = mean_stats.variance_sample();
		if (options.compute_law_of_total_variance) {
			return std::pair<num_t, num_t> (mean_stats.mean(), std::max<num_t>(0, var + var_stats.mean()) );
		}
		return std::pair<num_t, num_t> (mean_stats.mean(), std::max<num_t>(0, var) );
	}


	/* \brief predict the mean and the variance deviation for a configuration marginalized over a given set of partial configurations
	 * 
	 * This function will be mostly used to predict the mean over a given set of instances, but could be used to marginalize over any discrete set of partial configurations.
	 * 
	 * \param features a (partial) configuration where unset values should be set to NaN
	 * \param set_features a array containing the (partial) assignments used for the averaging. Every NaN value will be replaced by the corresponding value from features.
	 * \param set_size number of feature vectors in set_features
	 * 
	 * \return std::pair<num_t, num_t> mean and variance prediction of a feature vector averaged over 
	 */
    /*
	std::pair<num_t, num_t> predict_mean_var_marginalized_over_set (num_t *features, num_t* set_features, index_t set_size){
		
		num_t fv[num_features];

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> mean_stats, var_stats;
		for (auto i=0u; i < set_size; ++i){
			// construct the actual feature vector
			rfr::util::merge_two_vectors(features, &set_features[i*num_features], fv, num_features);

			num_t m , v;
			std::tie(m, v) = predict_mean_var(fv);

			mean_stats(m);
			var_stats(v);
		}
		return(std::pair<num_t, num_t> (mean_stats.mean(), std::max<num_t>(0, mean_stats.variance() + var_stats.mean()) ));
	}
    */

	/* \brief predict the mean and the variance of the mean prediction across a set of partial features
	 * 
	 * A very special function to predict the mean response of a a partial assignment for a given set.
	 * It takes the prediction of set-mean of every individual tree and combines to estimate the mean its
	 * total variance. The predictions of two trees are considered uncorrelated
	 * 
	 * \param features a (partial) configuration where unset values should be set to NaN
	 * \param set_features a 1d-array containing the (partial) assignments used for the averaging. Every NaN value will be replaced by the corresponding value from features. The array must hold set_size times the number of features entries! There is no consistency check!
	 * \param set_size number of feature vectors in set_features
	 * 
	 * \return std::tuple<num_t, num_t, num_t> mean and variance of empirical mean prediction of a feature vector averaged over. The last one is the estimated variance of a sample drawn from partial assignment.
	 */
    /*
	std::tuple<num_t, num_t, num_t> predict_mean_var_of_mean_response_on_set (num_t *features, num_t* set_features, index_t set_size){

			num_t fv[num_features];

			rfr::util::running_statistics<num_t> mean_stats, var_stats, sample_var_stats, sample_mean_stats;

			for (auto &t : the_trees){

					rfr::util::running_statistics<num_t> tree_mean_stats, tree_var_stats;

					for (auto i=0u; i < set_size; ++i){

							rfr::util::merge_two_vectors(features, &set_features[i*num_features], fv, num_features);

							num_t m , v; index_t n;
							std::tie(m, v, n) = t.predict_mean_var_N(fv);

							tree_mean_stats(m); tree_var_stats(v); sample_mean_stats(m); sample_var_stats(v);
					}

					mean_stats(tree_mean_stats.mean());
					var_stats(std::max<num_t>(0, tree_var_stats.mean()));
					
			}
			
			return(std::make_tuple(mean_stats.mean(), std::max<num_t>(0, mean_stats.variance()) + std::max<num_t>(0, var_stats.mean()/set_size), std::max<num_t>(0,sample_mean_stats.variance() + sample_var_stats.mean())));
	}
    */

	/* \brief estimates the covariance of two feature vectors
	 * 
	 * 
	 * The covariance between to input vectors contains information about the
	 * feature space. For other models, like GPs, this is a natural quantity
	 * (e.g., property of the kernel). Here, we try to estimate it using the
	 * emprical covariance of the individual tree's predictions.
	 * 
	 * \param f1 a valid feature vector (no sanity checks are performed!)
	 * \param f2 a second feature vector (no sanity checks are performed!)
	 */
    
	num_t covariance (const std::vector<num_t> &f1, const std::vector<num_t> &f2){

		rfr::util::running_covariance<num_t> run_cov_of_means;

		for (auto &t: the_trees)
			run_cov_of_means.push(t.predict(f1),t.predict(f2));

		return(run_cov_of_means.covariance());
	}



	/* \brief computes the kernel of a 'Kernel Random Forest'
	 *
	 * Source: "Random forests and kernel methods" by Erwan Scornet
	 * 
	 * The covariance between to input vectors contains information about the
	 * feature space. For other models, like GPs, this is a natural quantity
	 * (e.g., property of the kernel). Here, we try to estimate it using the
	 * emprical covariance of the individual tree's predictions.
	 * 
	 * \param f1 a valid feature vector (no sanity checks are performed!)
	 * \param f2 a second feature vector (no sanity checks are performed!)
	 */
    
	num_t kernel (const std::vector<num_t> &f1, const std::vector<num_t> &f2){

		rfr::util::running_statistics<num_t> stat;

		for (auto &t: the_trees){
			auto l1 = t.find_leaf_index(f1);
			auto l2 = t.find_leaf_index(f2);

			stat.push(l1==l2);
		}
		return(stat.mean());
	}



    
	std::vector< std::vector<num_t> > all_leaf_values (const std::vector<num_t> &feature_vector) const {
		std::vector< std::vector<num_t> > rv;
		rv.reserve(the_trees.size());

		for (auto &t: the_trees){
			rv.push_back(t.leaf_entries(feature_vector));
		}
		return(rv);
	}

	std::vector<tree_type> get_all_trees() const {return the_trees;}

	/* \brief given the feature vector and a given minimal number of points, find the nodes in each tree that all the
	 * subspaces represented by the union of the nodes contains the feature_vectorm further more the union of the
	 * subspaces should at least contain more points than num_points_minimal, we search from root nodes to leaves nodes
	 * */
	/* Problematic implementation TODO if the issue can be fixed
	std::vector<index_t> collect_data_nodes_from_root (const std::vector<num_t> &feature_vector, num_t num_points_minimal)
    {
	    assert(num_points_minimal <= num_data_points);
	    // the number of points contained in the complement set of the subspaces should be smaller than this one
        //nodes information in descend order
        struct node_info_des{
            index_t node_idx;
            index_t child_idx;
            bool is_leaf;
            // if we search from root towards leaves, it is better to use the complement of the node indices set
            std::set<num_t> sData_indices;
            std::set<num_t> sData_indices2shrink;
        };
        //typedef std::tuple<index_t, index_t, bool, std::set<num_t>, std::set<num_t>> node_info_des;

	    std::vector<index_t> vNode_indices;
	    std::vector<node_info_des> vNodes_info;
        vNodes_info.reserve(num_trees());
	    for (index_t i =0; i< the_trees.size(); ++i)
        {
	        const tree_type& tree = the_trees[i];
	        index_t node_id = 0;
	        const auto& node = tree.get_node(0);
	        bool is_a_leaf = node.is_a_leaf();
	        std::set<num_t> sData_indices;
	        for (index_t j =0; j< num_data_points; ++j)
            {
	            if(bootstrap_sample_weights[i][j] != 0)
                    sData_indices.insert(j);
            }
	        if (is_a_leaf)
            {
                node_info_des node_info(0, 0, is_a_leaf, sData_indices, std::set<index_t>());
                vNodes_info.push_back(node_info);
            }
	        else
            {
	            index_t child_idx = node.falls_into_child(feature_vector);
	            index_t child_idx_brother = 1 - child_idx; // brother of child node, notice this only works with binary tree TODO: improve this to fit polytrees
	            std::set<index_t> sData_indices2shrink(tree.get_data_indices_by_node(child_idx_brother));
                node_info_des node_info(0, child_idx, is_a_leaf, sData_indices, sData_indices2shrink);
                vNodes_info.push_back(node_info);
            }
        }

        std::set<index_t> sIntersect_indices(vNodes_info[0].sData_indices_comple);
        for (auto it_node_info = vNodes_info.begin() +1; it_node_info != vNodes_info.end(); ++it_node_info)
        {
            rfr::util::set_union(sIntersect_indices, (*it_node_info).sData_indices_comple);
        }

        if( sIntersect_indices.size() < num_points_minimal)
            return std::vector<index_t> (num_trees(), 0);

	    while (true)
        {
	        bool all_leaf = true;

            std::vector<std::pair<index_t, index_t>> num_indices2shrink;
            num_indices2shrink.reserve(num_trees());

	        for (index_t i=0; i < vNodes_info.size(); ++i)
            {
                node_info_des & node_info = vNodes_info[i];
                all_leaf &= node_info.is_leaf;
                std::pair<index_t, index_t> P = std::make_pair(node_info.sData_indices2shrink.size(), i);
                num_indices2shrink.push_back(P);
            }
	        if (all_leaf)
	            break; // if all the nodes are leaf node, simply return all the leaf node_idx

            //TODO consider here sorting in descending or ascending order
            // TODO some problem with the implementations. improve that later!!!
            std::sort(num_indices2shrink.begin(), num_indices2shrink.end(), std::greater<std::pair<index_t, index_t>>());
            for (auto it:num_indices2shrink)
            {
                index_t idx_next = it.second;
                node_info_des& node_info = vNodes_info[idx_next];
                if (node_info.is_leaf)
                    continue;
                rfr::util::set_difference(sIntersect_indices, node_info.sData_indices2shrink);
                if (sIntersect_indices.size() < num_points_minimal)
                    break;
                const tree_type& tree = the_trees[idx_next];
                node_info.node_idx = node_info.child_idx;
                const auto& node = tree.get_node(node_info.node_idx);
                node_info.child_idx = node.falls_into_child(feature_vector);
                node_info.is_leaf = node.is_a_leaf();
                rfr::util::set_difference(node_info.sData_indices, node_info.sData_indices2shrink);

                index_t child_idx_brother = 1 - node_info.child_idx; // brother of child node, notice this only works with binary tree TODO: improve this to fit polytrees
                std::set<index_t> sData_indices2shrink(tree.get_data_indices_by_node(child_idx_brother));
                node_info.sData_indices2shrink = sData_indices2shrink;
            }
        }

	    std::vector<index_t> node_indices;
        node_indices.reserve(num_trees());
        for (auto node_info: vNodes_info)
        {
            node_indices.push_back(std::get<0>(node_info));
        }
        return node_indices;
    }
    */

    /* \brief given the feature vector and a given minimal number of points, find the nodes in each tree that all the
 * subspaces represented by the union of the nodes contains the feature_vector further more the union of the
 * subspaces should at least contain more points than num_points_minimal, we search from leaves towards root
 * */
    std::vector<index_t> collect_data_nodes_from_leaf (const std::vector<num_t> &feature_vector, index_t num_points_minimal, index_t num_points_maximal)
    {
        assert(num_points_minimal <= num_data_points);
        // the number of points contained in the complement set of the subspaces should be smaller than this one
        //nodes information in descend order
        struct node_info_asc{
            const tree_type* pTree;
            index_t node_idx;
            index_t parent_idx;
            bool stop_update;
            // if we search from root towards leaves, it is better to use the complement of the node indices set
            std::set<index_t> sData_indices;
            std::set<index_t> sData_indices2expand;
        };

        std::vector<index_t> vNode_indices;
        std::vector<node_info_asc> vNodes_info;
        vNodes_info.reserve(num_trees());
        for (index_t i =0; i< num_trees(); ++i)
        {
            const tree_type* pTree = &the_trees[i];
            index_t node_id = pTree->find_leaf_index(feature_vector);
            const auto& node = pTree->get_node(node_id);
            bool stop_update = !bool(node_id);
            std::vector<index_t> vData_indices = pTree->get_data_indices_by_node(node_id);
            std::set<index_t> sData_indices(vData_indices.begin(), vData_indices.end());

            if (stop_update)
            {
                node_info_asc node_info={pTree, 0, 0, true, sData_indices, std::set<index_t>()};
                vNodes_info.push_back(node_info);
            }
            else
            {
                index_t parent_idx = node.get_parent_index();
                auto children_idx = pTree->get_node(parent_idx).get_children();
                index_t brother_idx = pTree->get_node(parent_idx).get_child_index(int(children_idx[0] == node_id));
                std::vector<index_t> vData_indices2expand = pTree->get_data_indices_by_node(brother_idx);
                std::set<index_t> sData_indices2expand(vData_indices2expand.begin(), vData_indices2expand.end());
                node_info_asc node_info= {pTree, node_id, parent_idx, false, sData_indices, sData_indices2expand};
                vNodes_info.push_back(node_info);
            }
        }

        std::set<index_t> sUnion_indices(vNodes_info[0].sData_indices);
        for (auto it_node_info = vNodes_info.begin() +1; it_node_info != vNodes_info.end(); ++it_node_info)
        {
            rfr::util::set_union(sUnion_indices, (*it_node_info).sData_indices);
        }

        if (sUnion_indices.size() <= num_points_minimal)
        {
            while (true)
            {
                bool stop_update_all = true;
                bool stop_iteration = false;

                std::vector<std::pair<index_t, index_t>> vData_indices2expand;
                vData_indices2expand.reserve(vNodes_info.size());

                for (index_t i=0; i < vNodes_info.size(); ++i)
                {
                    node_info_asc & node_info = vNodes_info[i];
                    if (!node_info.stop_update)
                    {
                        // the nodes that stop being updated will stop being added in the following terms
                        stop_update_all = false;
                        std::pair<index_t, index_t> P = std::make_pair(node_info.sData_indices2expand.size(), i);
                        vData_indices2expand.push_back(P);
                    }
                }
                if (stop_update_all)
                    break;

                vData_indices2expand.shrink_to_fit();
                //TODO consider here sorting in descending or ascending order
                std::sort(vData_indices2expand.begin(), vData_indices2expand.end(), std::greater<std::pair<index_t, index_t>>());
                /* TODO when two constraints conflicts, which threshold we should follow
                if (stop_update_all)
                {
                    index_t idx_next = vData_indices2expand.back().second;
                    node_info_asc& node_info = vNodes_info[idx_next];
                    node_info.node_idx = node_info.parent_idx;
                    break; // if all the nodes cannot be further updated, end loop
                }*/
                for (auto it:vData_indices2expand)
                {
                    index_t idx_next = it.second;
                    node_info_asc& node_info = vNodes_info[idx_next];
                    if (!node_info.stop_update) // just for double-check
                    {
                        std::set<index_t> sUnion_indices_look_ahead(sUnion_indices);
                        rfr::util::set_union(sUnion_indices_look_ahead, node_info.sData_indices2expand);

                        if (sUnion_indices_look_ahead.size() > num_points_maximal)
                        {
                            node_info.stop_update=true;
                        }
                        else if (sUnion_indices_look_ahead.size() >= num_points_minimal)
                        {
                            node_info.node_idx = node_info.parent_idx;
                            stop_iteration = true;
                            break;
                        }
                        else
                        {
                            sUnion_indices = sUnion_indices_look_ahead;
                            index_t parent_idx = node_info.parent_idx;
                            bool stop_update_next = !bool(parent_idx);
                            rfr::util::set_union(node_info.sData_indices, node_info.sData_indices2expand);
                            if (stop_update_next)
                            {
                                node_info.node_idx = parent_idx;
                                node_info.stop_update = true;
                                node_info.sData_indices2expand = std::set<index_t>();
                            }
                            else
                            {
                                const auto& parent = node_info.pTree->get_node(parent_idx);
                                index_t grandparent_idx = parent.get_parent_index();
                                auto uncle_and_parent_idices = node_info.pTree->get_node(grandparent_idx).get_children();
                                // brother of current node, only works with binary tree TODO: improve this to fit polytrees
                                index_t uncle_idx = node_info.pTree->get_node(grandparent_idx).get_child_index(int(uncle_and_parent_idices[0] == parent_idx));
                                std::vector<index_t> vData_indices2expand = node_info.pTree->get_data_indices_by_node(uncle_idx);
                                std::set<index_t> sData_indices2expand(vData_indices2expand.begin(), vData_indices2expand.end());

                                node_info.node_idx = parent_idx;
                                node_info.parent_idx = grandparent_idx;
                                node_info.sData_indices2expand = sData_indices2expand;
                            }
                        }
                    }
                }
                if (stop_iteration)
                    break;
            }
        }

        std::vector<index_t> node_indices;
        node_indices.reserve(num_trees());
        for (auto node_info: vNodes_info)
        {
            node_indices.push_back(node_info.node_idx);
        }
        return node_indices;
    }

	
	/* \brief returns the predictions of every tree marginalized over the NAN values in the feature_vector
	 * 
	 * TODO: more documentation over how the 'missing values' are handled
	 * 
	 * \param feature_vector non-specfied values (NaN) will be marginalized over according to the training data
	 */
	//std::vector<num_t> marginalized_mean_predictions(const std::vector<num_t> &feature_vector) const {
	//	std::vector<num_t> rv;
	//	rv.reserve(the_trees.size());
	//	for (auto &t : the_trees)
	//		rv.emplace_back(t.marginalized_mean_prediction(feature_vector));
	//	return(rv);
	//}



	/* \brief updates the forest by adding the provided datapoint without a complete retraining
	 * 
	 * 
	 * As retraining can be quite expensive, this function can be used to quickly update the forest
	 * by finding the leafs the datapoints belong into and just inserting them. This is, of course,
	 * not the right way to do it for many data points, but it should be a good approximation for a few.
	 * 
	 * \param features a valid feature vector
	 * \param response the corresponding response value
	 * \param weight the associated weight
	 */
	void pseudo_update (std::vector<num_t> features, response_t response, num_t weight){
		for (auto &t: the_trees)
			t.pseudo_update(features, response, weight);
	}
	
	/* \brief undoing a pseudo update by removing a point
	 * 
	 * This function removes one point from the corresponding leaves into
	 * which the given feature vector falls
	 * 
	 * \param features a valid feature vector
	 * \param response the corresponding response value
	 * \param weight the associated weight
	 */
	void pseudo_downdate(std::vector<num_t> features, response_t response, num_t weight){
		for (auto &t: the_trees)
			t.pseudo_downdate(features, response, weight);
	}
	
	num_t out_of_bag_error(){return(oob_error);}

	/* \brief writes serialized representation into a binary file
	 * 
	 * \param filename name of the file to store the forest in. Make sure that the directory exists!
	 */
	void save_to_binary_file(const std::string filename){
		std::ofstream ofs(filename, std::ios::binary);
		binary_oarch_t oarch(ofs);
		serialize(oarch);
	}

	/* \brief deserialize from a binary file created by save_to_binary_file
	 *
	 * \param filename name of the file in which the forest is stored. 
	 */
	void load_from_binary_file(const std::string filename){
		std::ifstream ifs(filename, std::ios::binary);
		binary_iarch_t iarch(ifs);
		serialize(iarch);
	}

	/* serialize into a string; used for Python's pickle.dump
	 * 
	 * \return std::string a JSON serialization of the forest
	 */
	std::string ascii_string_representation(){
		std::stringstream oss;
		{
			ascii_oarch_t oarch(oss);
			serialize(oarch);
		}
		return(oss.str());
	}

	/* \brief deserialize from string; used for Python's pickle.load
	 * 
	 * \return std::string a JSON serialization of the forest
	 */
	void load_from_ascii_string( std::string const &str){
		std::stringstream iss;
		iss.str(str);
		ascii_iarch_t iarch(iss);
		serialize(iarch);
	}



	/* \brief stores a latex document for every individual tree
	 * 
	 * \param filename_template a string to specify the location and the naming scheme. Note the directory is not created, so make sure it exists.
	 * 
	 */
	void save_latex_representation(const std::string filename_template){
		for (auto i = 0u; i<the_trees.size(); i++){
			std::stringstream filename;
			filename << filename_template<<i<<".tex";
			the_trees[i].save_latex_representation(filename.str().c_str());
		}
	}

	void print_info(){
		for (auto t: the_trees){
			t.print_info();
		}
	}


	virtual unsigned int num_trees (){ return(the_trees.size());}
	
};


}}//namespace rfr::forests
#endif

from collections import defaultdict
from itertools import cycle

import ipyvolume as ipv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from pandas import DataFrame
from pandas.plotting import parallel_coordinates
from scipy.stats import sem
from seaborn import color_palette
from sklearn import decomposition, metrics, preprocessing
from sklearn.cluster import KMeans

from ..util._io import df_to_csv, dict_to_pickle, read_groups_folders
from .api import Cell

_ALL_FEATURE_NAMES = (
    'surface_area',
    'total_length',
    'avg_process_thickness',
    'convex_hull',
    'no_of_forks',
    'no_of_primary_branches',
    'no_of_secondary_branches',
    'no_of_tertiary_branches',
    'no_of_quatenary_branches',
    'no_of_terminal_branches',
    'avg_length_of_primary_branches',
    'avg_length_of_secondary_branches',
    'avg_length_of_tertiary_branches',
    'avg_length_of_quatenary_branches',
    'avg_length_of_terminal_branches',
    'critical_radius',
    'critical_value',
    'enclosing_radius',
    'ramification_index',
    'skewness',
    'coefficient_of_determination',
    'sholl_regression_coefficient',
    'regression_intercept')

def _analyze_cells(groups_folders, img_type, shell_step_sz, save_features, show_logs):
    """Performs complete reading & feature extraction for groups of cells.

    Parameters
    ----------
    groups_folders : list
        A list of strings containing path of each folder with image dataset.
    img_type : str
        Neuroimaging technique used to get image data of neuronal cell,
        either 'confocal' or 'DAB'.
    shell_step_sz : int
        Difference (in pixels) between concentric Sholl circles, by default 3
    save_features : bool
        To save the features into a file.
    show_logs : bool
        To display logs of groups analysis.

    Returns
    -------
    features : DataFrame
        A DataFrame representing 23 morphometric of individual cells from
        each group.
    targets : list
        Representing Group ID of each Cell.
    sholl_original_plots : list
        Original values for Sholl radii vs. no. of intersections for each cell.
    sholl_polynomial_plots : list
        Estimated values for Sholl radii vs. no. of intersections for each cell.
    group_cnts : list
        A list representing counts of members (cells) in each group.
    file_names : list
        Names of each cell image file.

    """
    file_names, dataset = read_groups_folders(groups_folders)

    dataset_features, targets = [], []

    sholl_original_plots = []
    sholl_polynomial_plots = []

    group_cnts = []
    cell_cnt = 0

    for group_no, group in enumerate(dataset):
        group_cell_cnt = 0
        for cell_image in group:
            if show_logs:
                print(file_names[cell_cnt])

            cell_cnt += 1
            group_cell_cnt += 1

            targets.append(group_no)

            cell = Cell(cell_image, img_type, shell_step_size=shell_step_sz)
            cell_features = list(cell.features.values())

            sholl_original_plots.append(
                (cell._sholl_radii, cell._sholl_intersections))

            sholl_polynomial_plots.append((
                cell._non_zero_sholl_radii,
                cell._non_zero_sholl_intersections))

            dataset_features.append(cell_features)

        group_cnts.append(group_cell_cnt)

    features = DataFrame(dataset_features, columns=_ALL_FEATURE_NAMES)

    if save_features == True:
        FEATURES_DIR, FEATURES_FILE_NAME = '/Features/', 'features.csv'
        out_features = features.copy()
        out_features.insert(0, 'cell_image_file', file_names)
        df_to_csv(out_features, FEATURES_DIR, FEATURES_FILE_NAME)

    return (features, targets, sholl_original_plots, sholl_polynomial_plots,
            group_cnts, file_names)


class Groups:
    """Container object for groups of cells of nervous system
    
    It extract 23 Morphometric features of all the cells in each group &
    provides group level sholl analysis, PCA & clustering according to
    those features.

    Parameters
    ----------
    groups_folders : list
        Path to folders containing input cell images with each path
        corresponding to different subgroups.
    image_type : str
        Neuroimaging technique used to get image data of all cells,
        either 'confocal' or 'DAB'. This assumes that the group datasets are
        of homogeneous `image_type`.
    labels : dict
        Group labels to be used for visualization. Specify for each group.
    shell_step_size : int, optional
        Difference (in pixels) between concentric Sholl circles, by default 3
    save_features : bool, optional
        To save a file containing morphometric features of each cell,
        by default True
    show_logs : bool, optional
        To show logs of analysis of each cell, by default False

    Attributes
    ----------
    features : DataFrame
        A DataFrame representing 23 morphometric of individual cells from
        each group.
    shell_step_size : int
        Difference (in pixels) between concentric Sholl circles, by default 3
    file_names : list
        Names of each cell image file.
    targets : list
        Representing Group ID of each Cell.
    group_counts : list
        A list representing counts of members (cells) in each group.
    sholl_original_plots : list
        Original values for Sholl radii vs. no. of intersections for each cell.
    sholl_polynomial_plots : list
        Estimated values for Sholl radii vs. no. of intersections for each cell.
    labels : dict
        Labels for each group to be used for visualization.
    markers : dict
        marker for each group to be used in visualization.
    feature_significance : ndarray
        Coefficient values for each feature used for calculating the PCA,
        represents the significance of the feature.
    projected : ndarray
        Coordinates of each Cell in the multidimensional space defined by the
        Principal Components.

    """
    __slots__ = ('features', 'targets', 'sholl_original_plots', 'labels',
                 'sholl_polynomial_plots', 'pca_feature_names', 'markers',
                 'feature_significance', 'file_names', 'group_counts',
                 'projected', 'shell_step_size')
    def __init__(self, groups_folders, image_type, labels, shell_step_size=3, save_features=True, show_logs=False):
        self.labels = labels
        self.shell_step_size = shell_step_size

        (
            self.features,
            self.targets,
            self.sholl_original_plots,
            self.sholl_polynomial_plots, 
            self.group_counts,
            self.file_names
        ) = _analyze_cells(groups_folders, image_type,
                           shell_step_size, save_features, show_logs)

        self.pca_feature_names = self.markers = self.feature_significance = None

    def plot_avg_sholl_plot(self, save_results=True):
        """Plots average Sholl Plot

        Parameters
        ----------
        save_results : bool, optional
            To save a file containing Sholl Plots for each cell, by default True
        """
        ANALYSES = ['single_cell_intersections', 'group_cells_intersections']
        SINGLE_CELL_INTERSECTIONS = ['original_plot', 'polynomial_plot']

        largest_radius = []
        no_of_intersections = []
        write_buffer = {ANALYSES[0]: {}, ANALYSES[1]: {}}
        file_names = self.file_names
        shell_step_size = self.shell_step_size
        original_plots = self.sholl_original_plots
        polynomial_plots = self.sholl_polynomial_plots
        group_cnts = self.group_counts
        labels = self.labels

        for index in range(len(file_names)):
            write_buffer[ANALYSES[0]][file_names[index]] = {
                SINGLE_CELL_INTERSECTIONS[0]: original_plots[index],
                SINGLE_CELL_INTERSECTIONS[1]: polynomial_plots[index]}

            # get the max radius of each cell, as smallest and
            # mid-level ones can be inferred from shell_step_size
            largest_radius.append(max(polynomial_plots[index][0]))
            no_of_intersections.append(polynomial_plots[index][1])

        for group_no, cnt in enumerate(group_cnts):
            group_cnt = sum(group_cnts[:group_no+1])
            group_radius = max(largest_radius[group_cnt-cnt:group_cnt])

            current_intersections = no_of_intersections[group_cnt -
                                                        cnt:group_cnt]
            current_radiuses = range(
                shell_step_size, group_radius+1, shell_step_size)

            intersection_dict = defaultdict(list)
            for intersections in current_intersections:
                for i, intersection_val in enumerate(intersections):
                    intersection_dict[current_radiuses[i]].append(
                        intersection_val)
            write_buffer[ANALYSES[1]][group_no] = intersection_dict

        if save_results:
            DIR, OUTFILE = '/Sholl Results/', 'sholl_results.pickle'
            dict_to_pickle(write_buffer, DIR, OUTFILE)

        for group_no, group_sholl in write_buffer[ANALYSES[1]].items():
            x, y, e = [], [], []
            for radius, intersections in group_sholl.items():
                x.append(radius)
                intersections += group_cnts[group_no] * [0]
                intersections = intersections[:group_cnts[group_no]]
                y.append(np.mean(intersections))
                e.append(sem(intersections))
            plt.errorbar(x, y, yerr=e, label=labels[group_no])

        plt.xlabel("Distance from soma")
        plt.ylabel("No. of intersections")
        plt.legend()
        plt.show()
    
    def pca(self, n_PC, color_dict, markers, on_features=None, save_results=True):
        """Principal Component Analysis of morphological features of cells.

        Parameters
        ----------
        n_PC : int
            If greater than 1, return n_PC number of Principal Components after
            clustering. If None & use_features is False, it's autoselected as
            number of Principal Components calculated.
        color_dict : dict
            Dict with color to be used for each group.
        marker : dict
            Dict with marker for each group to be used in PCA plot.
        on_features : list, optional
            List of names of morphological features from which Principal
            Components will be derived, by default None.
            If None, all 23 morphological features will be used.
        save_results : bool, optional
            To save a file containing PCA values, by default True

        Returns
        -------
        var_PCs : ndarray
            Captured variance ratios of each Principal Component.

        Raises
        ------
        ValueError
            * If n_PC isn't greater than 1 & less than the total number of
            morphological features of cells.
            * If element(s) of on_features is/are not in list of all
            morphological features.

        """
        all_features = list(_ALL_FEATURE_NAMES)
        targets = self.targets
        labels = self.labels

        if n_PC < 2 or n_PC >= len(all_features):
            raise ValueError('Principal Components must be greater than 1 & '
                            'less than number of morphological features.')

        if on_features == None:
            on_features = all_features
        elif not all(feature in all_features for feature in set(on_features)):
            raise ValueError('Selected Principal Components are not a subset '
                             'of the original set of morphological features.')

        self.pca_feature_names = on_features
        self.markers = markers

        def get_cov_ellipse(cov, centre, nstd, **kwargs):
            """Return an Ellipse patch representing the covariance matrix
            `cov` centred at `centre` and scaled by the factor `nstd`.

            """
            # Find and sort eigenvalues and eigenvectors into descending order
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]

            # The anti-clockwise angle to rotate our ellipse by
            vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
            theta = np.arctan2(vy, vx)

            # Width and height of ellipse to draw
            width, height = nstd * np.sqrt(eigvals)

            return Ellipse(centre, width, height, np.degrees(theta), **kwargs)

        subset_features = self.features[on_features].to_numpy()

        pca_object = decomposition.PCA(n_PC)

        # Scale data
        scaler = preprocessing.MaxAbsScaler()
        scaler.fit(subset_features)
        X = scaler.transform(subset_features)

        # fit on data
        pca_object.fit(X)

        # access values and vectors
        feature_significance = pca_object.components_

        # variance captured by principal components
        var_PCs = pca_object.explained_variance_ratio_

        # transform data
        projected = pca_object.transform(X)

        PC_1 = projected[:, 0]
        PC_2 = projected[:, 1]

        if save_results:
            PC_COLUMN_NAMES = [f'PC {itr + 1}' for itr in range(n_PC)]
            pca_values = DataFrame(data=projected, columns=PC_COLUMN_NAMES)
            df_to_csv(pca_values, '/PCA Results/', 'pca values.csv')

        def visualize_two_PCs():
            n_std = 3  # no. of standard deviations to show
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('white')
            for l in np.unique(targets):
                ix = np.where(targets == l)
                mean_PC_1 = np.mean(PC_1[ix])
                mean_PC_2 = np.mean(PC_2[ix])
                cov = np.cov(PC_1, PC_2)
                ax.scatter(PC_1[ix], PC_2[ix], c=color_dict[l], s=40,
                            label=labels[l], marker=markers[l])
                e = get_cov_ellipse(cov, (mean_PC_1, mean_PC_2),
                                    n_std, fc=color_dict[l], alpha=0.4)
                ax.add_artist(e)

            plt.xlabel(f'PC 1 (Variance: {var_PCs[0]*100:.1f})', fontsize=14)
            plt.ylabel(f'PC 2 (Variance: {var_PCs[1]*100:.1f})', fontsize=14)
            plt.legend(title='Groups')
            plt.show()

        visualize_two_PCs()

        self.feature_significance = feature_significance
        self.projected = projected

        return var_PCs

    def plot_feature_histograms(self, features=list(_ALL_FEATURE_NAMES)):
        """Plots feature significance heatmap.

        Raises
        ------
        ValueError
            If `features` is not a subset of 23 Morphometric features.

        """
        if not set(features).issubset(features):
            raise ValueError('Given feature names must be a subset or equal to '
                             'set of 23 Morphometric features.')

        axes = plt.subplots((len(features)+1)//2, 2, figsize=(15, 12))[1]
        data = self.features[features].to_numpy()

        ko = data[np.where(np.array(self.targets) == 0)[0]]
        control = data[np.where(np.array(self.targets) == 1)[0]]
        ax = axes.ravel()  # flat axes with numpy ravel

        for i in range(len(features)):
            bins = np.histogram(data[:, i], bins=40)[1]
            # red color for malignant class
            ax[i].hist(ko[:, i], bins=bins, color='r', alpha=.5)
            # alpha is for transparency in the overlapped region
            ax[i].hist(control[:, i], bins=bins, color='g', alpha=0.3)
            ax[i].set_title(features[i], fontsize=9)
            # x-axis co-ordinates aren't so useful, as we just want to
            # look how well separated the histograms are
            ax[i].axes.get_xaxis().set_visible(False)
            ax[i].set_yticks(())

        ax[0].legend(self.markers, loc='best', fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_feature_significance_heatmap(self):
        """Plots feature significance heatmap.

        Raises
        ------
        RuntimeError
            If Principal Components are not found before plotting
            feature significance heatmap.

        """
        if not hasattr(self, 'projected'):
            raise RuntimeError('Principal Components must be found before '
                               'plotting feature significance heatmap.')
        n_PC = self.feature_significance.shape[0]
        feature_significance = self.feature_significance
        significance_order_PC_1 = np.argsort(feature_significance[0])
        sorted_feature_significance = np.zeros(feature_significance.shape)

        def order_by_significance(significance):
            out = np.array(significance)[significance_order_PC_1]
            return out

        sorted_feature_significance = list(map(
            order_by_significance, feature_significance))

        sorted_feature_names = np.array(self.pca_feature_names)[
            significance_order_PC_1]

        plt.matshow(np.array(sorted_feature_significance), cmap='gist_heat')
        plt.yticks(list(range(n_PC)), [
                    f'PC {i+1}' for i in range(n_PC)], fontsize=10)
        plt.colorbar(orientation='horizontal')
        plt.xticks(range(len(sorted_feature_names)),
                    sorted_feature_names, rotation=65, ha='left')
        plt.show()

    def plot_feature_significance_vectors(self):
        """Plots feature significance vectors after PCA.

        Raises
        ------
        RuntimeError
            If Principal Components are not found before plotting
            feature significance vectors.

        """
        if not hasattr(self, 'projected'):
            raise RuntimeError('Principal Components must be found before '
                               'plotting feature significance vectors.')
        score = self.projected
        coeff = np.transpose(self.feature_significance)
        labels = self.pca_feature_names
        xs = score[:, 0]
        ys = score[:, 1]
        n = coeff.shape[0]
        scale_x = 1.0 / (xs.max() - xs.min())
        scale_y = 1.0 / (ys.max() - ys.min())

        plt.figure(figsize=(10, 9))
        ax = plt.axes()
        ax.scatter(xs * scale_x, ys * scale_y, c=self.targets, alpha=.9)
        for i in range(n):
            ax.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
            if labels is None:
                ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                        f'Var{i+1}', color='g', ha='center', va='center')
            else:
                ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                        labels[i], color='g', ha='center', va='center')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        plt.show()

    def get_clusters(self, k=None, use_features=True, n_PC=None, plot='parallel'):
        """
        Highly configurable K-Means clustering & visualization of cell data.

        Parameters
        ----------
        k : int or None, default None
            If greater than 1, return k number of clusters. Else if None it is
            autoselected, on basis of maximum Calinski Harabasz Score.
        use_features : bool, default True
            If True, clustering would use original set of morphometric features.
        n_PC : int or None, default None
            If greater than 1, return n_PC number of Principal Components after
            clustering. If None & use_features is False, it's autoselected as
            number of Principal Components calculated.
        plot : str or None, default 'parallel'
            The type of plot user would like to get, either parallel or scatter.

        Returns
        -------
        centers_df
            A DataFrame with normalized center coordinates of each cluster.
        df
            A DataFrame with normalized cell coordinates & the respective
            cluster to which they belong.

        """
        all_features = self.features
        group_cnts = self.group_counts
        labels = iter(self.labels.values())
        markers = iter(self.markers.values())
        n_cells = all_features.shape[0]
        seed = np.random.randint(0, n_cells)  # for deterministic randomness

        if k != None and (k < 2 or k > n_cells):
            raise ValueError('Number of clusters, k, must be greater than 1 & '
                                'lesser than the total number of cells.')

        if plot not in [None, 'parallel', 'scatter']:
            raise ValueError('Plot must be either of parallel or scatter.')

        def compute_clusters(n_clusters, normalized_features, max_clusters):
            best_variance_ratio, best_k, best_model = 0, 0, None
            K = range(2, max_clusters +
                        1) if n_clusters == None else [n_clusters]

            # Either autoselect least varying K-Means model or use specified k
            for k in K:
                # convergence is costly, better to prespecify k
                model = KMeans(n_clusters=k, random_state=seed).fit(
                    normalized_features)
                variance_ratio = metrics.calinski_harabasz_score(
                    normalized_features, model.labels_)
                if variance_ratio > best_variance_ratio:
                    best_variance_ratio = variance_ratio
                    best_model = model
                    best_k = k

            return best_k, best_model, best_variance_ratio

        if use_features:
            if n_PC == None:
                scaler = preprocessing.MaxAbsScaler()
                features = all_features.to_numpy()
                df = DataFrame(scaler.fit(features).transform(features))
            else:
                raise ValueError('Cannot use morphological features & n_PC '
                                    'simultaneously')
        else:
            if not hasattr(self, 'projected'):
                raise RuntimeError('Principal Components must be found before '
                                    'clustering in their feature space.')

            projected = self.projected

            if n_PC is None:
                # autoselect n_PC as number of computed Principal Components
                n_PC = projected.shape[1]

            if 1 < n_PC <= projected.shape[1]:
                PC_COLUMN_NAMES = [f'PC {itr}' for itr in range(1, n_PC+1)]
                df = DataFrame(projected[:, :n_PC], columns=PC_COLUMN_NAMES)
            else:
                raise ValueError('Number of Principal Components, n_PC, should '
                                 'be greater than 1 & less than or equal to the'
                                 ' total number of Principal Components.')

        if n_PC not in [2, 3] and plot == 'scatter':
            raise ValueError('Scatter plot can only be in 2D or 3D.')

        normalized_feature_vector = df.to_numpy()

        # TODO: Select best max value for k
        k, kmeans_model, variance_ratio = compute_clusters(
            k, normalized_feature_vector, 8)

        # creates a dataframe with a column for cluster number
        centers_df = DataFrame(kmeans_model.cluster_centers_,
                               columns=PC_COLUMN_NAMES)
        centers_df['cluster'] = range(k)

        LABEL_COLOR_MAP = color_palette(None, k)

        def parallel_plot(data):
            plt.figure(figsize=(15, 8)).gca().axes.set_ylim([-3, 3])
            parallel_coordinates(data, 'cluster',
                                    color=LABEL_COLOR_MAP, marker='o')

        def scatter_plot(data, centers):
            l_idx = r_idx = 0
            label_color = [LABEL_COLOR_MAP[l] for l in kmeans_model.labels_]

            if data.shape[1] == 2:
                for i in range(k):
                    plt.text(centers[i, 0], centers[i, 1], i, weight='bold',
                                size=10, backgroundcolor=LABEL_COLOR_MAP[i],
                                color='white')

                for cells in group_cnts:
                    r_idx += cells
                    plt.scatter(data[l_idx:r_idx, 0], data[l_idx:r_idx, 1], 40,
                                label_color[l_idx:r_idx], next(markers),
                                label=next(labels), alpha=.9)
                    l_idx += cells

                plt.xlabel('PC1'), plt.ylabel('PC2')
                plt.legend(title='Groups')
                plt.show()
            elif data.shape[1] == 3:  # ipyvolume 3D scatter plot
                # assuming 2 classes: stab & ctrl
                MARKERS = cycle(['box', 'sphere', 'arrow', 'point_2d',
                                    'square_2d', 'triangle_2d', 'circle_2d'])
                ipv.figure()
                ipv.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                            LABEL_COLOR_MAP, 5, 5.6, marker='diamond')

                for cells in group_cnts:
                    r_idx += cells
                    ipv.scatter(data[l_idx:r_idx, 0], data[l_idx:r_idx, 1],
                                data[l_idx:r_idx, 2], label_color[l_idx:r_idx],
                                4, 4.6, marker=next(MARKERS))
                    l_idx += cells

                ipv.xyzlabel('PC1', 'PC2', 'PC3')
                ipv.show()

        if plot != None:
            if plot == 'parallel':
                parallel_plot(centers_df)
            elif plot == 'scatter':
                scatter_plot(normalized_feature_vector,
                                kmeans_model.cluster_centers_)

        print(f'k = {k} clusters (0, ..., {k})',
              f'with Variance Ratio = {variance_ratio}')
        print(f'seed = {seed}')
        print('Using', ('principal components',
                        'morphometric features')[use_features])

        df['cluster'] = kmeans_model.labels_

        return centers_df, df
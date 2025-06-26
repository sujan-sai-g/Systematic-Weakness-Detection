from .core import SliceLineR, SliceLineRplus, NaivePPIestimator


def run_three_approaches(
    df, 
    gt_columns, 
    clip_columns, 
    clip_precisions, 
    label_map,
    level=2, 
    alpha=0.95, 
    top_k=100,
    err_col_name = "error",
    inv_threshold = 0.1
):
    # Ground Truth slices
    slicelineGT = SliceLineR(df, gt_columns, lvl=level, err_column=err_col_name)
    slices_gt = (
        slicelineGT.update_level(level)
        .update_alpha(alpha)
        .top_slices(top_k)
    )

    # CLIP slices
    slicelineClip = SliceLineR(df, clip_columns, lvl=level, err_column=err_col_name)
    slices_clip = (
        slicelineClip.update_level(level)
        .update_alpha(alpha)
        .top_slices(top_k)
    )

    # Corrected slices
    prec_estimator = NaivePPIestimator(df, clip_columns, clip_precisions, label_map, inv_thresh=inv_threshold)
    slicelineCorrected = SliceLineRplus(
        df, clip_columns, errorEstimator=prec_estimator, lvl=level, err_column=err_col_name
    )
    slices_corrected = (
        slicelineCorrected.update_level(level)
        .update_alpha(alpha)
        .top_slices(top_k)
    )

    return slices_gt, slices_clip, slices_corrected


def run_two_approaches(
    df, 
    clip_columns, 
    clip_precisions, 
    label_map,
    level=2, 
    alpha=0.95, 
    top_k=100,
    err_col_name = "error",
    inv_threshold = 0.1
):
    
    # CLIP slices
    slicelineClip = SliceLineR(df, clip_columns, lvl=level, err_column=err_col_name)
    slices_clip = (
        slicelineClip.update_level(level)
        .update_alpha(alpha)
        .top_slices(top_k)
    )

    # Corrected slices
    prec_estimator = NaivePPIestimator(df, clip_columns, clip_precisions, label_map, inv_thresh=inv_threshold)
    slicelineCorrected = SliceLineRplus(
        df, clip_columns, errorEstimator=prec_estimator, lvl=level,err_column=err_col_name
    )
    slices_corrected = (
        slicelineCorrected.update_level(level)
        .update_alpha(alpha)
        .top_slices(top_k)
    )

    return slices_clip, slices_corrected


def clean_slice_results(slices_gt=None, slices_clip=None, slices_corrected=None, error_rate=None):
    """
    Clean and standardize slice results.
    
    Args:
        slices_gt: Ground truth slices (optional)
        slices_clip: CLIP slices (optional)
        slices_corrected: Corrected slices (optional)
        
    Returns:
        tuple: Cleaned DataFrames
    """
    results = []
    
    if slices_gt is not None:
        slices_gt = slices_gt.copy()
        slices_gt.drop(["lvl"], axis=1, inplace=True, errors='ignore')
        slices_gt = slices_gt[slices_gt["score"] > 0]
        slices_gt.rename(columns={
            "score": "slice_score", 
            "eRate": "slice_average_error", 
            "n": "slice_size"
        }, inplace=True)
        if error_rate is not None:
            slices_gt = slices_gt[slices_gt["slice_average_error"] > 1.5 * error_rate]
        results.append(slices_gt)
    
    if slices_clip is not None:
        slices_clip = slices_clip.copy()
        slices_clip.drop(["lvl"], axis=1, inplace=True, errors='ignore')
        slices_clip = slices_clip[slices_clip["score"] > 0]
        slices_clip.rename(columns={
            "score": "slice_score", 
            "eRate": "slice_average_error", 
            "n": "slice_size"
        }, inplace=True)
        if error_rate is not None:
            slices_clip = slices_clip[slices_clip["slice_average_error"] > 1.5 * error_rate]
        results.append(slices_clip)
    
    if slices_corrected is not None:
        slices_corrected = slices_corrected.copy()
        slices_corrected.drop(["lvl"], axis=1, inplace=True, errors='ignore')
        slices_corrected = slices_corrected[slices_corrected["scoreC"] > 0]
        slices_corrected.rename(columns={
            "scoreC": "slice_score", 
            "eRateC": "slice_average_error", 
            "nC": "slice_size"
        }, inplace=True)
        if error_rate is not None:
            slices_corrected = slices_corrected[slices_corrected["slice_average_error"] > 1.5 * error_rate]
        results.append(slices_corrected)
        
    
    return tuple(results) if len(results) > 1 else results[0]
from opensfm import io
from opensfm import reconstruction
from opensfm.dataset_base import DataSetBase


def run_dataset(data: DataSetBase) -> None:
    """Compute the SfM reconstruction."""

    tracks_manager = data.load_tracks_manager()
    algorithm = data.config.get('reconstruction_algorithm', 'incremental')

    if algorithm == 'incremental':
        report, reconstructions = reconstruction.incremental_reconstruction(
            data, tracks_manager
        )
    elif algorithm == 'triangulation':
        report, reconstructions = reconstruction.triangulation_reconstruction(
            data, tracks_manager
        )
    elif algorithm == 'planar':
        report, reconstructions = reconstruction.planar_reconstruction(
            data, tracks_manager
        )
    else:
        raise RuntimeError(f"Unsupported algorithm for reconstruction {algorithm}")

    data.save_reconstruction(reconstructions)
    data.save_report(io.json_dumps(report), "reconstruction.json")

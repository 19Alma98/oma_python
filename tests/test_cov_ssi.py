import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from dynoma.constants import BASE_DTYPE
from dynoma.exceptions import ModalIdentificationError
from dynoma.cov_ssi import CovSSI
from dynoma.utils import compute_mac_value

TOLERANCE = 1e-3
cov_ssi_algorithm = CovSSI(
    frequency_max=5,
    frequency_min=0,
    order_max=5,
    order_min=1,
    order_steps=1,
    min_mpc=0.03,
    time_lag=1.2,
    number_of_fft_points=2**4,
    num_svd_plots=2,
    damping_max_value=10,
    frequency_noise_threshold=0.03,
    damping_noise_threshold=0.03,
    mac_noise_threshold=0.03,
    minimum_cluster_dimension=1,
    maximum_distance=0.03,
    continuous_mode=False,
)


def test_reorder_poles_should_return_expected_results(cov_ssi_not_clustered_modal_parameters):
    input_frequencies, input_damping_ratios, input_mode_shapes, input_orders = cov_ssi_not_clustered_modal_parameters

    expected_frequencies = np.array([[4.03, 20.76, 20.04], [5.17, 0.0, 18.3]])
    expected_damping = np.array([[3.95, 4.72, 1.83], [5.74, 0.0, 0.67]])
    expected_modes = np.array(
        [
            [
                [4.13e-05 - 2.02e-04j, -2.57e-04 + 4.14e-04j, -6.91e-05 + 1.41e-04j],
                [3.29e-05 + 2.03e-04j, 0.00e00 + 0.00e00j, 2.88e-05 + 7.15e-05j],
            ],
            [
                [-2.19e-04 + 1.18e-03j, 3.09e-05 - 1.68e-05j, -1.80e-04 + 5.66e-05j],
                [-9.19e-06 - 3.35e-04j, 0.00e00 + 0.00e00j, 3.61e-05 - 6.65e-05j],
            ],
            [
                [2.51e-05 - 1.21e-04j, 6.96e-05 - 7.52e-05j, 1.03e-04 + 1.49e-07j],
                [9.86e-06 - 4.80e-05j, 0.00e00 + 0.00e00j, -4.47e-05 - 2.98e-05j],
            ],
            [
                [-1.96e-06 - 7.48e-07j, 3.18e-05 - 5.61e-05j, -1.29e-05 + 2.24e-06j],
                [-7.36e-05 - 2.27e-04j, 0.00e00 + 0.00e00j, -7.19e-06 - 1.37e-05j],
            ],
            [
                [1.35e-04 - 8.93e-04j, 1.58e-03 - 1.15e-03j, 1.04e-03 + 8.04e-05j],
                [-1.00e-05 - 1.15e-04j, 0.00e00 + 0.00e00j, -3.68e-04 - 2.62e-04j],
            ],
            [
                [6.22e-05 - 3.23e-04j, 5.97e-04 - 4.21e-04j, 4.04e-04 + 3.58e-05j],
                [1.06e-05 - 1.30e-04j, 0.00e00 + 0.00e00j, -1.43e-04 - 1.26e-04j],
            ],
            [
                [4.48e-06 - 2.02e-05j, 3.70e-05 - 3.70e-05j, 2.15e-05 - 5.43e-05j],
                [-5.28e-07 - 1.70e-04j, 0.00e00 + 0.00e00j, 2.82e-05 - 1.85e-07j],
            ],
            [
                [-7.72e-06 - 4.42e-06j, 3.22e-05 + 3.06e-05j, -7.05e-07 + 5.55e-05j],
                [-3.73e-05 + 2.89e-04j, 0.00e00 + 0.00e00j, 1.62e-06 + 2.61e-05j],
            ],
        ]
    )
    steps = np.arange(40, 80 + 20, 20)
    oma_parameters = cov_ssi_algorithm._reorder_poles(
        frequencies=input_frequencies,
        damping_ratios=input_damping_ratios,
        mode_shapes=input_mode_shapes,
        model_orders=input_orders,
        steps=steps,
    )
    np.testing.assert_allclose(oma_parameters["frequencies"], expected_frequencies, atol=TOLERANCE)
    np.testing.assert_allclose(oma_parameters["damping_ratios"], expected_damping, atol=TOLERANCE)
    np.testing.assert_allclose(oma_parameters["mode_shapes"], expected_modes, atol=TOLERANCE)


def test_get_dashboard_graph_data_should_return_expected_results(
    cov_ssi_not_clustered_modal_parameters,
):
    input_freq, input_damping, input_modes, input_orders = cov_ssi_not_clustered_modal_parameters
    steps = np.arange(40, 80 + 20, 20)
    oma_parameters = cov_ssi_algorithm._reorder_poles(
        frequencies=input_freq,
        damping_ratios=input_damping,
        mode_shapes=input_modes,
        model_orders=input_orders,
        steps=steps,
    )
    reordered_poles_parameters = cov_ssi_algorithm._stability_poles_analysis(
        input_frequencies=oma_parameters["frequencies"],
        input_damping_ratios=oma_parameters["damping_ratios"],
        input_modes=oma_parameters["mode_shapes"],
    )

    frequency_by_stability, model_order_by_stability = cov_ssi_algorithm._create_dashboard_graph_data(
        stability_results=reordered_poles_parameters, model_orders=steps
    )

    expected_frequency_dict = {
        "new_pole": [4.03, 4.03, 5.17, 5.17, 20.76, 20.76],
        "stable_pole": [],
        "stable_frequency_and_mac": [],
        "stable_frequency_and_damping_ratios": [],
        "stable_frequency": [],
    }

    expected_orders_dict = {
        "new_pole": [40.0, 40.0, 40.0, 40.0, 60.0, 60.0],
        "stable_pole": [],
        "stable_frequency_and_mac": [],
        "stable_frequency_and_damping_ratios": [],
        "stable_frequency": [],
    }

    for key in [
        "new_pole",
        "stable_pole",
        "stable_frequency_and_mac",
        "stable_frequency_and_damping_ratios",
        "stable_frequency",
    ]:
        np.testing.assert_allclose(expected_frequency_dict[key], frequency_by_stability[key], atol=TOLERANCE)
        np.testing.assert_allclose(expected_orders_dict[key], model_order_by_stability[key], atol=TOLERANCE)


@pytest.mark.parametrize(
    "input_array",
    [
        np.array([0, 1, 2]),
        np.array([]),
        np.array([[[1, 2, 3], [4, 5, 6]]]),
    ],
)
def test_compute_impulse_response_should_raise_error_if_input_not_bi_dimensional(
    input_array,
):
    with pytest.raises(
        ModalIdentificationError,
        match=f"Input signals must be 2-dimensional, given input with {len(input_array.shape)} dimensions",
    ):
        cov_ssi_algorithm._compute_impulse_response(signal=input_array, time_step=0.5)


@pytest.mark.parametrize(
    "input_array",
    [
        np.array([0, 1, 2]),
        np.array([]),
        np.array([[[1, 2, 3], [4, 5, 6]]]),
    ],
)
def test_compute_impulse_response_opt_should_raise_error_if_input_not_bi_dimensional(
    input_array,
):
    with pytest.raises(
        ModalIdentificationError,
        match=f"Input signals must be 2-dimensional, given input with {len(input_array.shape)} dimensions",
    ):
        cov_ssi_algorithm._compute_impulse_response_optimized(signal=input_array, time_step=0.5)


@pytest.mark.parametrize(
    "input_array, expected_irf",
    [
        (
            np.array([[1, 3, 5, 7, 9, 11], [0, 2, 4, 6, 8, 10]]),
            np.array(
                [
                    [
                        [11.66666667, 4.81102495, 0.47236655, -2.05613229, -3.3469524],
                        [11.66666667, 4.81102495, 0.47236655, -2.05613229, -3.3469524],
                    ],
                    [
                        [11.66666667, 4.81102495, 0.47236655, -2.05613229, -3.3469524],
                        [11.66666667, 4.81102495, 0.47236655, -2.05613229, -3.3469524],
                    ],
                ]
            ),
        ),
        (
            np.array([[1, 0], [3, 2], [5, 4], [7, 6], [9, 10], [11, 12]]),
            np.array(
                [
                    [
                        [11.66666667, 4.81102495, 0.47236655, -2.05613229, -3.3469524],
                        [14.33333333, 5.72741066, -0.15745552, -2.70543723, -3.94196616],
                    ],
                    [
                        [14.33333333, 6.36888065, 1.10218862, -2.27256727, -4.53697992],
                        [17.88888889, 7.59072826, 0.36739621, -3.06616219, -5.3303316],
                    ],
                ]
            ),
        ),
    ],
)
def test_compute_correlation_should_return_expected_array(input_array, expected_irf):
    actual_irf = cov_ssi_algorithm._compute_impulse_response(signal=input_array, time_step=0.5)

    np.testing.assert_allclose(expected_irf, actual_irf, atol=TOLERANCE)


def test_compute_correlation_should_return_expected_array_with_number_of_channels_equal_to_2():
    """Same IRF golden policy as parametrized correlation test."""
    input_array = np.array([[1, 3, 5, 7, 9, 11], [0, 2, 4, 6, 8, 10]])
    expected_irf = np.array(
        [
            [
                [11.66666667, 4.81102495, 0.47236655, -2.05613229, -3.3469524],
                [11.66666667, 4.81102495, 0.47236655, -2.05613229, -3.3469524],
            ],
            [
                [11.66666667, 4.81102495, 0.47236655, -2.05613229, -3.3469524],
                [11.66666667, 4.81102495, 0.47236655, -2.05613229, -3.3469524],
            ],
        ]
    )
    actual_irf = cov_ssi_algorithm._compute_impulse_response(signal=input_array, time_step=0.5)

    np.testing.assert_allclose(expected_irf, actual_irf, atol=TOLERANCE)


def test__build_hankel_matrix_should_return_expected_arrays():
    input_array = np.array([[[1, 3, 5], [7, 9, 11]], [[0, 2, 4], [6, 8, 10]]])
    hankel_matrix = cov_ssi_algorithm._build_hankel_matrix(input_array)
    tensor_hankel_matrix = torch.tensor(hankel_matrix)
    u, s, _ = torch.linalg.svd(tensor_hankel_matrix)

    expected_u = np.array([[-0.7548654989621743, -0.6558796219403277], [-0.6558796219403278, 0.7548654989621745]])
    expected_s = np.array([12.560725391999933, 0.47767941840536315])
    np.testing.assert_allclose(expected_u, u, atol=TOLERANCE)
    np.testing.assert_allclose(expected_s, s, atol=TOLERANCE)


def test_modal_identification_should_raise_error_if_lstsq_raises_error(
    mocker: MockerFixture,
):
    mock_lstsq = mocker.patch("dynoma.cov_ssi.torch.linalg.lstsq")
    mock_lstsq.side_effect = ValueError
    u = torch.tensor(
        [
            [-0.33075111, 0.86229048, -0.38348249],
            [-0.91904106, -0.38662796, -0.0766965],
            [-0.21439972, 0.32706871, 0.92035799],
        ]
    )

    s = torch.tensor([1.72209340e01, 1.85457066e00, 2.86932054e-16])

    with pytest.raises(ModalIdentificationError, match="Modal identification cannot be performed."):
        cov_ssi_algorithm._perform_modal_identification(
            u=u, s=s, number_of_channels=2, time_step=1 / 10, number_of_steps=3
        )


def test_modal_identification_should_return_expected_arrays():
    """Golden arrays for torch lstsq/eigen path; sensitive to pinned torch/BLAS (atol=TOLERANCE)."""
    u = torch.tensor(
        [
            [-0.33075111, 0.86229048, -0.38348249],
            [-0.91904106, -0.38662796, -0.0766965],
            [-0.21439972, 0.32706871, 0.92035799],
        ]
    )

    s = torch.tensor([1.72209340e01, 1.85457066e00, 2.86932054e-16])

    expected_frequencies = np.array(
        [
            [1.02505279, 0.59401286, 0.59401268, 0.59401268, 0.59401268],
            [0.0, 5.06349182, 5.06349182, 5.06349182, 5.06349182],
            [0.0, 0.0, 64.53679657, 64.53679657, 64.53679657],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected_damping_ratios = np.array(
        [
            [99.99999237, 100.0, 100.0, 100.0, 100.0],
            [0.0, 15.78631878, 15.78631878, 15.78631878, 15.78631878],
            [0.0, 0.0, 99.69942474, 99.69942474, 99.69942474],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    actual_frequencies, actual_damping_ratios, _ = cov_ssi_algorithm._perform_modal_identification(
        u=u, s=s, number_of_channels=2, time_step=1 / 10, number_of_steps=5
    )
    np.testing.assert_allclose(expected_frequencies, actual_frequencies, atol=TOLERANCE)
    np.testing.assert_allclose(expected_damping_ratios, actual_damping_ratios, atol=TOLERANCE)


@pytest.mark.parametrize(
    "mode_shape_1, mode_shape_2, mac_value",
    [
        (
            np.array(
                [
                    -2.50155790272220e-06 - 3.25601943797008e-06j,
                    -0.000515547396818131 - 0.000516218487420379j,
                    -0.000960236589309833 - 0.000460088370616296j,
                    2.81774450565676e-05 - 0.000100310505827090j,
                    -0.00862364101195196 - 0.00420980172323539j,
                    -0.00308157353238295 - 0.00179158707534183j,
                    1.14026829002432e-05 + 8.36838247510715e-05j,
                    4.51689147044627e-05 - 1.73247105973111e-05j,
                ]
            ),
            np.array(
                [
                    -0.000561492302540442 - 7.48736716664220e-05j,
                    -0.000237221128475168 + 0.000987864898356125j,
                    -0.000185225301544795 + 0.000220627961673206j,
                    -3.93467820735593e-05 + 9.83497937507037e-05j,
                    -0.00217835683667423 + 0.00134431724959743j,
                    -0.000759000499279380 + 0.000343874726238863j,
                    -0.000107923865062948 - 2.06266752865337e-05j,
                    -7.08144101334980e-05 + 0.000201435020508188j,
                ]
            ),
            0.8493415879505946,
        ),
        (
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            np.array(
                [
                    -0.000561492302540442 - 7.48736716664220e-05j,
                    -0.000237221128475168 + 0.000987864898356125j,
                    -0.000185225301544795 + 0.000220627961673206j,
                    -3.93467820735593e-05 + 9.83497937507037e-05j,
                    -0.00217835683667423 + 0.00134431724959743j,
                    -0.000759000499279380 + 0.000343874726238863j,
                    -0.000107923865062948 - 2.06266752865337e-05j,
                    -7.08144101334980e-05 + 0.000201435020508188j,
                ]
            ),
            0.0,
        ),
    ],
)
def test_get_stability_mac_value_should_return_expected_values(mode_shape_1, mode_shape_2, mac_value):
    actual_mac_value = compute_mac_value(mode_shape_1, mode_shape_2)
    np.testing.assert_allclose(mac_value, actual_mac_value, atol=TOLERANCE)


@pytest.mark.parametrize(
    "frequency_stability, damping_stability, mode_shape_stability, expected_stability_status",
    [(0, 1, 1, 0), (1, 1, 1, 1), (1, 0, 1, 2), (1, 1, 0, 3), (1, 0, 0, 4)],
)
def test_get_stability_pole_status_should_return_expected_values(
    frequency_stability,
    damping_stability,
    mode_shape_stability,
    expected_stability_status,
):
    actual_stability_status = cov_ssi_algorithm._get_stability_pole_status(
        frequency_stability=frequency_stability,
        damping_stability=damping_stability,
        mode_shape_stability=mode_shape_stability,
    )
    assert actual_stability_status == expected_stability_status


def test_oma_stability_check_should_return_expected_results(
    oma_stability_check_inputs,
):
    freq, damping, mode_shape = oma_stability_check_inputs

    expected_frequencies = np.array([13.1286698984175] * 4 + [15.4745993292697] * 12)

    expected_damping_ratios = np.array(
        [59.7387429022622] * 4
        + [38.4229244077966] * 2
        + [0.000200000000000000]
        + [38.4229244077966] * 2
        + [0.000200000000000000] * 2
        + [38.4229244077966] * 3
        + [0.000200000000000000]
        + [38.4229244077966]
    )
    expected_mode_shape = np.array(
        [
            [0 + 0j] * 16,
            [
                -0.00023722113110125065 - 0.0009878649143502116j,
                -0.00023722113110125065 - 0.0009878649143502116j,
                -0.00023722113110125065 - 0.0009878649143502116j,
                -0.00023722113110125065 - 0.0009878649143502116j,
                -0.00023722113110125065 + 0.0009878649143502116j,
                -0.0005155474063940346 - 0.000516218482516706j,
                -0.0005155474063940346 + 0.000516218482516706j,
                -0.0005155474063940346 - 0.000516218482516706j,
                -0.00023722113110125065 + 0.0009878649143502116j,
                -0.0005155474063940346 + 0.000516218482516706j,
                -0.0005155474063940346 + 0.000516218482516706j,
                -0.0005155474063940346 - 0.000516218482516706j,
                -0.00023722113110125065 + 0.0009878649143502116j,
                -0.0005155474063940346 - 0.000516218482516706j,
                -0.0005155474063940346 + 0.000516218482516706j,
                -0.00023722113110125065 + 0.0009878649143502116j,
            ],
            [
                -0.0001852253044489771 - 0.00022062796051613986j,
                -0.0001852253044489771 - 0.00022062796051613986j,
                -0.0001852253044489771 - 0.00022062796051613986j,
                -0.0001852253044489771 - 0.00022062796051613986j,
                -0.0001852253044489771 + 0.00022062796051613986j,
                -0.0009602365898899734 - 0.00046008836943656206j,
                -0.0009602365898899734 + 0.00046008836943656206j,
                -0.0009602365898899734 - 0.00046008836943656206j,
                -0.0001852253044489771 + 0.00022062796051613986j,
                -0.0009602365898899734 + 0.00046008836943656206j,
                -0.0009602365898899734 + 0.00046008836943656206j,
                -0.0009602365898899734 - 0.00046008836943656206j,
                -0.0001852253044489771 + 0.00022062796051613986j,
                -0.0009602365898899734 - 0.00046008836943656206j,
                -0.0009602365898899734 + 0.00046008836943656206j,
                -0.0001852253044489771 + 0.00022062796051613986j,
            ],
            [
                -3.9346781704807654e-05 - 9.834979573497549e-05j,
                -3.9346781704807654e-05 - 9.834979573497549e-05j,
                -3.9346781704807654e-05 - 9.834979573497549e-05j,
                -3.9346781704807654e-05 - 9.834979573497549e-05j,
                -3.9346781704807654e-05 + 9.834979573497549e-05j,
                2.817744461935945e-05 - 0.00010031050624093041j,
                2.817744461935945e-05 + 0.00010031050624093041j,
                2.817744461935945e-05 - 0.00010031050624093041j,
                -3.9346781704807654e-05 + 9.834979573497549e-05j,
                2.817744461935945e-05 + 0.00010031050624093041j,
                2.817744461935945e-05 + 0.00010031050624093041j,
                2.817744461935945e-05 - 0.00010031050624093041j,
                -3.9346781704807654e-05 + 9.834979573497549e-05j,
                2.817744461935945e-05 - 0.00010031050624093041j,
                2.817744461935945e-05 + 0.00010031050624093041j,
                -3.9346781704807654e-05 + 9.834979573497549e-05j,
            ],
            [
                -0.0021783567499369383 - 0.001344317221082747j,
                -0.0021783567499369383 - 0.001344317221082747j,
                -0.0021783567499369383 - 0.001344317221082747j,
                -0.0021783567499369383 - 0.001344317221082747j,
                -0.0021783567499369383 + 0.001344317221082747j,
                -0.008623640984296799 - 0.004209801554679871j,
                -0.008623640984296799 + 0.004209801554679871j,
                -0.008623640984296799 - 0.004209801554679871j,
                -0.0021783567499369383 + 0.001344317221082747j,
                -0.008623640984296799 + 0.004209801554679871j,
                -0.008623640984296799 + 0.004209801554679871j,
                -0.008623640984296799 - 0.004209801554679871j,
                -0.0021783567499369383 + 0.001344317221082747j,
                -0.008623640984296799 - 0.004209801554679871j,
                -0.008623640984296799 + 0.004209801554679871j,
                -0.0021783567499369383 + 0.001344317221082747j,
            ],
            [
                -0.0007590004825033247 - 0.0003438747371546924j,
                -0.0007590004825033247 - 0.0003438747371546924j,
                -0.0007590004825033247 - 0.0003438747371546924j,
                -0.0007590004825033247 - 0.0003438747371546924j,
                -0.0007590004825033247 + 0.0003438747371546924j,
                -0.0030815736390650272 - 0.0017915870994329453j,
                -0.0030815736390650272 + 0.0017915870994329453j,
                -0.0030815736390650272 - 0.0017915870994329453j,
                -0.0007590004825033247 + 0.0003438747371546924j,
                -0.0030815736390650272 + 0.0017915870994329453j,
                -0.0030815736390650272 + 0.0017915870994329453j,
                -0.0030815736390650272 - 0.0017915870994329453j,
                -0.0007590004825033247 + 0.0003438747371546924j,
                -0.0030815736390650272 - 0.0017915870994329453j,
                -0.0030815736390650272 + 0.0017915870994329453j,
                -0.0007590004825033247 + 0.0003438747371546924j,
            ],
            [
                -0.00010792386456159875 + 2.0626675905077718e-05j,
                -0.00010792386456159875 + 2.0626675905077718e-05j,
                -0.00010792386456159875 + 2.0626675905077718e-05j,
                -0.00010792386456159875 + 2.0626675905077718e-05j,
                -0.00010792386456159875 - 2.0626675905077718e-05j,
                1.1402682503103279e-05 + 8.36838225950487e-05j,
                1.1402682503103279e-05 - 8.36838225950487e-05j,
                1.1402682503103279e-05 + 8.36838225950487e-05j,
                -0.00010792386456159875 - 2.0626675905077718e-05j,
                1.1402682503103279e-05 - 8.36838225950487e-05j,
                1.1402682503103279e-05 - 8.36838225950487e-05j,
                1.1402682503103279e-05 + 8.36838225950487e-05j,
                -0.00010792386456159875 - 2.0626675905077718e-05j,
                1.1402682503103279e-05 + 8.36838225950487e-05j,
                1.1402682503103279e-05 - 8.36838225950487e-05j,
                -0.00010792386456159875 - 2.0626675905077718e-05j,
            ],
            [
                -7.081440708134323e-05 - 0.0002014350175159052j,
                -7.081440708134323e-05 - 0.0002014350175159052j,
                -7.081440708134323e-05 - 0.0002014350175159052j,
                -7.081440708134323e-05 - 0.0002014350175159052j,
                -7.081440708134323e-05 + 0.0002014350175159052j,
                4.516891567618586e-05 - 1.7324709915556014e-05j,
                4.516891567618586e-05 + 1.7324709915556014e-05j,
                4.516891567618586e-05 - 1.7324709915556014e-05j,
                -7.081440708134323e-05 + 0.0002014350175159052j,
                4.516891567618586e-05 + 1.7324709915556014e-05j,
                4.516891567618586e-05 + 1.7324709915556014e-05j,
                4.516891567618586e-05 - 1.7324709915556014e-05j,
                -7.081440708134323e-05 + 0.0002014350175159052j,
                4.516891567618586e-05 - 1.7324709915556014e-05j,
                4.516891567618586e-05 + 1.7324709915556014e-05j,
                -7.081440708134323e-05 + 0.0002014350175159052j,
            ],
        ]
    )

    expected_mac_values = np.array(
        [
            0.9011409282684326,
            0.88152676820755,
            1.0,
            0.7647690773010254,
            0.88152676820755,
            0.995161235332489,
            0.995161235332489,
            1.0,
            0.9011409282684326,
            1.0,
            0.9011409282684326,
            0.88152676820755,
            1.0,
            0.9011409282684326,
            0.88152676820755,
            0.7647690773010254,
        ],
        dtype=BASE_DTYPE,
    )
    expected_stability_status = np.array([0, 0, 1, 0, 3, 2, 2, 1, 4, 1, 4, 3, 1, 0, 0, 0])
    actual_stability_status_oma = cov_ssi_algorithm._oma_stability_check(
        frequencies_1=freq,
        damping_ratio_1=damping,
        mode_shape_1=mode_shape,
        frequencies_2=freq,
        damping_ratio_2=damping,
        mode_shape_2=mode_shape,
    )

    np.testing.assert_array_equal(expected_stability_status, actual_stability_status_oma["stability_status"])
    np.testing.assert_allclose(expected_frequencies, actual_stability_status_oma["frequencies"], atol=TOLERANCE)
    np.testing.assert_allclose(expected_damping_ratios, actual_stability_status_oma["damping_ratios"], atol=TOLERANCE)
    np.testing.assert_allclose(expected_mode_shape, actual_stability_status_oma["mode_shapes"], atol=TOLERANCE)
    np.testing.assert_allclose(expected_mac_values, actual_stability_status_oma["mac_values"], atol=TOLERANCE)


def test_stability_poles_analysis_should_return_expected_results(
    oma_poles_analysis_inputs,
):
    input_freq, input_damping_ratios, input_modes = oma_poles_analysis_inputs
    expected_stability_status = [np.array([1, 0, 0, 0, 0, 0, 2, 0]), np.array([2, 0, 0, 0, 4, 0, 0, 0])]
    expected_frequency = [
        np.array([4.02747361333968] * 4 + [5.22322516170052] * 4),
        np.array([4.035585554202902] * 4 + [5.17063349965044] * 4),
    ]

    expected_damping_ratios = [
        np.array([3.77096735188619] * 4 + [6.10331903304641] * 4),
        np.array([3.94604736517127] * 4 + [5.73867362594326] * 4),
    ]

    expected_modes = [
        np.array(
            [
                [
                    4.0961156628327444e-05 - 0.0001969492295756936j,
                    4.0961156628327444e-05 - 0.0001969492295756936j,
                    4.0961156628327444e-05 - 0.0001969492295756936j,
                    4.0961156628327444e-05 - 0.0001969492295756936j,
                    4.132512185606174e-05 + 0.00020264751219656318j,
                    4.132512185606174e-05 + 0.00020264751219656318j,
                    4.132512185606174e-05 + 0.00020264751219656318j,
                    4.132512185606174e-05 + 0.00020264751219656318j,
                ],
                [
                    -0.00023680480080656707 + 0.001148464740253985j,
                    -0.00023680480080656707 + 0.001148464740253985j,
                    -0.00023680480080656707 + 0.001148464740253985j,
                    -0.00023680480080656707 + 0.001148464740253985j,
                    -5.5514636187581345e-05 - 0.0002718115283641964j,
                    -5.5514636187581345e-05 - 0.0002718115283641964j,
                    -5.5514636187581345e-05 - 0.0002718115283641964j,
                    -5.5514636187581345e-05 - 0.0002718115283641964j,
                ],
                [
                    2.807624332490377e-05 - 0.00011938163515878841j,
                    2.807624332490377e-05 - 0.00011938163515878841j,
                    2.807624332490377e-05 - 0.00011938163515878841j,
                    2.807624332490377e-05 - 0.00011938163515878841j,
                    1.1416116649343167e-05 - 5.787194459117018e-05j,
                    1.1416116649343167e-05 - 5.787194459117018e-05j,
                    1.1416116649343167e-05 - 5.787194459117018e-05j,
                    1.1416116649343167e-05 - 5.787194459117018e-05j,
                ],
                [
                    -1.7209184477451345e-07 - 8.292745974358695e-07j,
                    -1.7209184477451345e-07 - 8.292745974358695e-07j,
                    -1.7209184477451345e-07 - 8.292745974358695e-07j,
                    -1.7209184477451345e-07 - 8.292745974358695e-07j,
                    -7.26951111573726e-05 - 0.00023642851738259196j,
                    -7.26951111573726e-05 - 0.00023642851738259196j,
                    -7.26951111573726e-05 - 0.00023642851738259196j,
                    -7.26951111573726e-05 - 0.00023642851738259196j,
                ],
                [
                    0.00016122944361995906 - 0.0008688477100804448j,
                    0.00016122944361995906 - 0.0008688477100804448j,
                    0.00016122944361995906 - 0.0008688477100804448j,
                    0.00016122944361995906 - 0.0008688477100804448j,
                    -4.9801587920228485e-06 - 0.00019894515571650118j,
                    -4.9801587920228485e-06 - 0.00019894515571650118j,
                    -4.9801587920228485e-06 - 0.00019894515571650118j,
                    -4.9801587920228485e-06 - 0.00019894515571650118j,
                ],
                [
                    7.214279321487993e-05 - 0.00031435172422789037j,
                    7.214279321487993e-05 - 0.00031435172422789037j,
                    7.214279321487993e-05 - 0.00031435172422789037j,
                    7.214279321487993e-05 - 0.00031435172422789037j,
                    1.5040038306324277e-05 - 0.0001664162118686363j,
                    1.5040038306324277e-05 - 0.0001664162118686363j,
                    1.5040038306324277e-05 - 0.0001664162118686363j,
                    1.5040038306324277e-05 - 0.0001664162118686363j,
                ],
                [
                    3.073174639212084e-06 - 2.2073823856771924e-05j,
                    3.073174639212084e-06 - 2.2073823856771924e-05j,
                    3.073174639212084e-06 - 2.2073823856771924e-05j,
                    3.073174639212084e-06 - 2.2073823856771924e-05j,
                    3.814346882791142e-06 - 0.00017163071606773883j,
                    3.814346882791142e-06 - 0.00017163071606773883j,
                    3.814346882791142e-06 - 0.00017163071606773883j,
                    3.814346882791142e-06 - 0.00017163071606773883j,
                ],
                [
                    -4.868051291850861e-06 + 1.7817519619711675e-06j,
                    -4.868051291850861e-06 + 1.7817519619711675e-06j,
                    -4.868051291850861e-06 + 1.7817519619711675e-06j,
                    -4.868051291850861e-06 + 1.7817519619711675e-06j,
                    -4.082172017660923e-05 + 0.00028695049695670605j,
                    -4.082172017660923e-05 + 0.00028695049695670605j,
                    -4.082172017660923e-05 + 0.00028695049695670605j,
                    -4.082172017660923e-05 + 0.00028695049695670605j,
                ],
            ]
        ),
        np.array(
            [
                [
                    4.1356892324984074e-05 - 0.00020231866801623255j,
                    4.1356892324984074e-05 - 0.00020231866801623255j,
                    4.1356892324984074e-05 - 0.00020231866801623255j,
                    4.1356892324984074e-05 - 0.00020231866801623255j,
                    3.297054718132131e-05 + 0.00020332906569819897j,
                    3.297054718132131e-05 + 0.00020332906569819897j,
                    3.297054718132131e-05 + 0.00020332906569819897j,
                    3.297054718132131e-05 + 0.00020332906569819897j,
                ],
                [
                    -0.0002196130808442831 + 0.0011864369735121727j,
                    -0.0002196130808442831 + 0.0011864369735121727j,
                    -0.0002196130808442831 + 0.0011864369735121727j,
                    -0.0002196130808442831 + 0.0011864369735121727j,
                    -9.195151505991817e-06 - 0.0003352410567458719j,
                    -9.195151505991817e-06 - 0.0003352410567458719j,
                    -9.195151505991817e-06 - 0.0003352410567458719j,
                    -9.195151505991817e-06 - 0.0003352410567458719j,
                ],
                [
                    2.510750164219644e-05 - 0.00012181690544821322j,
                    2.510750164219644e-05 - 0.00012181690544821322j,
                    2.510750164219644e-05 - 0.00012181690544821322j,
                    2.510750164219644e-05 - 0.00012181690544821322j,
                    9.862873412203044e-06 - 4.808191442862153e-05j,
                    9.862873412203044e-06 - 4.808191442862153e-05j,
                    9.862873412203044e-06 - 4.808191442862153e-05j,
                    9.862873412203044e-06 - 4.808191442862153e-05j,
                ],
                [
                    -1.968050810319255e-06 - 7.480221597688796e-07j,
                    -1.968050810319255e-06 - 7.480221597688796e-07j,
                    -1.968050810319255e-06 - 7.480221597688796e-07j,
                    -1.968050810319255e-06 - 7.480221597688796e-07j,
                    -7.369803643086925e-05 - 0.00022712261124979705j,
                    -7.369803643086925e-05 - 0.00022712261124979705j,
                    -7.369803643086925e-05 - 0.00022712261124979705j,
                    -7.369803643086925e-05 - 0.00022712261124979705j,
                ],
                [
                    0.00013590423623099923 - 0.0008932355558499694j,
                    0.00013590423623099923 - 0.0008932355558499694j,
                    0.00013590423623099923 - 0.0008932355558499694j,
                    0.00013590423623099923 - 0.0008932355558499694j,
                    -1.0099468454427551e-05 - 0.00011546654423000291j,
                    -1.0099468454427551e-05 - 0.00011546654423000291j,
                    -1.0099468454427551e-05 - 0.00011546654423000291j,
                    -1.0099468454427551e-05 - 0.00011546654423000291j,
                ],
                [
                    6.221151852514595e-05 - 0.00032361724879592657j,
                    6.221151852514595e-05 - 0.00032361724879592657j,
                    6.221151852514595e-05 - 0.00032361724879592657j,
                    6.221151852514595e-05 - 0.00032361724879592657j,
                    1.0615373867040034e-05 - 0.0001309944927925244j,
                    1.0615373867040034e-05 - 0.0001309944927925244j,
                    1.0615373867040034e-05 - 0.0001309944927925244j,
                    1.0615373867040034e-05 - 0.0001309944927925244j,
                ],
                [
                    4.488975264393957e-06 - 2.0212362869642675e-05j,
                    4.488975264393957e-06 - 2.0212362869642675e-05j,
                    4.488975264393957e-06 - 2.0212362869642675e-05j,
                    4.488975264393957e-06 - 2.0212362869642675e-05j,
                    -5.282099664327689e-07 - 0.0001709758653305471j,
                    -5.282099664327689e-07 - 0.0001709758653305471j,
                    -5.282099664327689e-07 - 0.0001709758653305471j,
                    -5.282099664327689e-07 - 0.0001709758653305471j,
                ],
                [
                    -7.728409400442615e-06 - 4.421136964083416e-06j,
                    -7.728409400442615e-06 - 4.421136964083416e-06j,
                    -7.728409400442615e-06 - 4.421136964083416e-06j,
                    -7.728409400442615e-06 - 4.421136964083416e-06j,
                    -3.7356796383392066e-05 + 0.0002895077341236174j,
                    -3.7356796383392066e-05 + 0.0002895077341236174j,
                    -3.7356796383392066e-05 + 0.0002895077341236174j,
                    -3.7356796383392066e-05 + 0.0002895077341236174j,
                ],
            ]
        ),
    ]

    stable_analysis = cov_ssi_algorithm._stability_poles_analysis(
        input_frequencies=input_freq, input_damping_ratios=input_damping_ratios, input_modes=input_modes
    )

    np.testing.assert_array_equal(np.array(expected_stability_status), np.array(stable_analysis["stability_status"]))
    np.testing.assert_allclose(np.array(expected_frequency), np.array(stable_analysis["frequencies"]), atol=TOLERANCE)
    np.testing.assert_allclose(
        np.array(expected_damping_ratios), np.array(stable_analysis["damping_ratios"]), atol=TOLERANCE
    )
    np.testing.assert_allclose(np.array(expected_modes), np.array(stable_analysis["mode_shapes"]), atol=TOLERANCE)


def test_clustering_analysis_should_return_expected_results(
    oma_clustering_analysis_inputs,
):
    frequencies, damping_ratios, mode_shapes = oma_clustering_analysis_inputs
    expected_frequencies = np.array([13.3170641850000, 18.3090731200000])
    expected_damping_ratios = np.array([1.87371760000000, 0.665719160000000])
    expected_mode_shapes = np.array(
        [
            [0.0342960612550388, 0.170535949409828],
            [0.0597403744751316, -0.167474017815051],
            [0.119046669269268, -0.118940923625027],
            [0.0205562892597795, -0.0342625975692678],
            [1, -1],
            [0.369020396050608, -0.422966711943501],
            [0.00511408381819312, 0.0623841556531359],
            [-0.00824802257262612, 0.0578322445516530],
        ]
    )
    expected_mode_shapes_complex = np.array(
        [
            [6.0228663200000004e-05 - 7.291538315e-06j, 2.88504789e-05 + 7.15172675e-05j],
            [7.117686793e-05 - 0.00011586282815j, 3.61854092e-05 - 6.652854e-05j],
            [0.0001712063065 - 0.0001829373825j, -4.47334219e-05 - 2.98634087e-05j],
            [3.0067387199999997e-05 + 1.1287546779999999e-05j, -7.19946901e-06 - 1.37194451e-05j],
            [0.0012968760244999999 - 0.0016770789649999999j, -0.000368060227 - 0.000262718917j],
            [0.0004958324035 - 0.0006046290995j, -0.000143565402 - 0.000126381681j],
            [8.84004602e-06 + 7.075716109999999e-06j, 2.82098312e-05 - 1.85450533e-07j],
            [-1.121701938e-05 - 9.305539594999999e-06j, 1.62134112e-06 + 2.61017355e-05j],
        ]
    )
    expected_cluster_dimensions = np.array([2, 2])
    expected_frequency_bounds = np.array([[13.0433142680000, 13.5908141020000], [18.3090731200000, 18.3090731200000]])
    expected_damping_bounds = np.array([[0.7549023836000002, 2.9925328164], [0.66571916, 0.66571916]])

    clustered_avt_parameters = cov_ssi_algorithm._cluster_modal_parameters(
        frequencies=frequencies,
        damping_ratios=damping_ratios,
        mode_shapes=mode_shapes,
        shuffle=False,
    )

    np.testing.assert_allclose(expected_frequencies, clustered_avt_parameters["frequencies"], atol=TOLERANCE)
    np.testing.assert_allclose(expected_damping_ratios, clustered_avt_parameters["damping_ratios"], atol=TOLERANCE)
    np.testing.assert_allclose(expected_mode_shapes, clustered_avt_parameters["real_mode_shapes"], atol=TOLERANCE)
    np.testing.assert_allclose(
        expected_mode_shapes_complex, clustered_avt_parameters["complex_mode_shapes"], atol=TOLERANCE
    )
    np.testing.assert_array_equal(expected_cluster_dimensions, clustered_avt_parameters["cluster_dimensions"])
    np.testing.assert_allclose(expected_frequency_bounds, clustered_avt_parameters["frequency_bounds"], atol=TOLERANCE)
    np.testing.assert_allclose(expected_damping_bounds, clustered_avt_parameters["damping_bounds"], atol=TOLERANCE)


def test_cut_off_modal_parameters_should_raise_error_if_no_frequencies_are_found(mocker: MockerFixture):
    starting_freq = np.array(
        [
            [20.60149642, 20.60714496],
            [17.98357885, 18.20341701],
            [17.03772016, 17.68461956],
            [14.68287526, 16.94396167],
            [13.73128503, 14.30982869],
        ]
    )
    starting_damping_ratios = np.array(
        [
            [1.85193639, 1.81796949],
            [2.2386566, 2.90767225],
            [2.20124449, 2.52176972],
            [3.39373821, 2.36504159],
            [2.96149542, 2.76309934],
        ]
    )
    starting_modes = np.array(
        [
            [
                [1.15292664e-05 + 9.93356550e-06j, -1.16573733e-05 - 9.69428708e-06j],
                [-2.09786628e-06 - 4.48244393e-06j, 4.36248615e-07 + 3.42422464e-06j],
                [3.01742914e-06 - 2.09219752e-06j, 1.37680708e-07 - 1.18516506e-06j],
                [1.79883360e-06 + 3.32832992e-06j, -2.70839830e-06 + 2.07084851e-06j],
                [9.30543646e-06 - 1.57842919e-05j, 9.90147315e-07 + 3.09510189e-06j],
            ],
            [
                [-6.03891001e-06 - 4.81107032e-06j, 7.10678992e-06 + 5.25519153e-06j],
                [-3.02599411e-06 - 5.46216288e-06j, 5.94678067e-06 - 5.64775605e-07j],
                [1.20071882e-05 + 7.70896485e-06j, 4.07310342e-06 - 7.73063408e-06j],
                [-2.75364439e-06 - 6.97883003e-07j, -1.37005791e-05 - 4.72710859e-06j],
                [-3.72024455e-06 + 1.51853591e-05j, 1.48135290e-06 + 1.74831593e-06j],
            ],
        ]
    )
    mock_non_zero = mocker.patch("dynoma.cov_ssi.np.nonzero")
    mock_non_zero.return_value = np.array([[]])
    with pytest.raises(
        ModalIdentificationError,
        match="No frequencies found with thresholds: freq_min=0.0, freq_max=5.0, damp_max=10.0",
    ):
        cov_ssi_algorithm._cut_off_modal_parameters(
            frequencies=starting_freq,
            damping_ratios=starting_damping_ratios,
            mode_shapes=starting_modes,
            number_of_steps=1,
        )


def test_cut_off_modal_parameters_should_raise_error_if_no_frequencies_are_found_after_phase_collinearity(
    mocker: MockerFixture,
):
    starting_freq = np.array(
        [
            [20.60149642, 20.60714496],
            [17.98357885, 18.20341701],
            [17.03772016, 17.68461956],
            [14.68287526, 16.94396167],
            [13.73128503, 14.30982869],
        ]
    )
    starting_damping_ratios = np.array(
        [
            [1.85193639, 1.81796949],
            [2.2386566, 2.90767225],
            [2.20124449, 2.52176972],
            [3.39373821, 2.36504159],
            [2.96149542, 2.76309934],
        ]
    )
    starting_modes = np.array(
        [
            [
                [1.15292664e-05 + 9.93356550e-06j, -1.16573733e-05 - 9.69428708e-06j],
                [-2.09786628e-06 - 4.48244393e-06j, 4.36248615e-07 + 3.42422464e-06j],
                [3.01742914e-06 - 2.09219752e-06j, 1.37680708e-07 - 1.18516506e-06j],
                [1.79883360e-06 + 3.32832992e-06j, -2.70839830e-06 + 2.07084851e-06j],
                [9.30543646e-06 - 1.57842919e-05j, 9.90147315e-07 + 3.09510189e-06j],
            ],
            [
                [-6.03891001e-06 - 4.81107032e-06j, 7.10678992e-06 + 5.25519153e-06j],
                [-3.02599411e-06 - 5.46216288e-06j, 5.94678067e-06 - 5.64775605e-07j],
                [1.20071882e-05 + 7.70896485e-06j, 4.07310342e-06 - 7.73063408e-06j],
                [-2.75364439e-06 - 6.97883003e-07j, -1.37005791e-05 - 4.72710859e-06j],
                [-3.72024455e-06 + 1.51853591e-05j, 1.48135290e-06 + 1.74831593e-06j],
            ],
        ]
    )
    mock_non_zero = mocker.patch("dynoma.cov_ssi.np.nonzero")
    mock_non_zero.return_value = np.array([[0, 1]])
    mock_delete = mocker.patch("dynoma.cov_ssi.np.delete")
    mock_delete.return_value = np.array([])
    with pytest.raises(
        ModalIdentificationError,
        match="No frequencies found with thresholds: freq_min=0.0, freq_max=5.0, damp_max=10.0, min_mpc=0.03",
    ):
        cov_ssi_algorithm._cut_off_modal_parameters(
            frequencies=starting_freq,
            damping_ratios=starting_damping_ratios,
            mode_shapes=starting_modes,
            number_of_steps=1,
        )


def test_stability_poles_analysis_should_return_empty_dict_when_index_error_is_raised():
    stable_analysis = cov_ssi_algorithm._stability_poles_analysis(
        input_frequencies=np.array([[0.0, 0.0, 0.0]]),
        input_damping_ratios=np.array([[0.0, 0.0, 0.0]]),
        input_modes=np.array([[[0.0, 0.0, 0.0]]]),
    )

    assert len(stable_analysis["frequencies"][1]) == 0


@pytest.mark.parametrize(
    "input_signals",
    [np.array([]), np.array([0, 1]), np.array([[[1, 2, 3]], [[4, 5, 6]]])],
)
def test_cov_ssi_should_raise_error_if_input_not_well_defined(input_signals):
    with pytest.raises(
        ModalIdentificationError,
        match="Input signals must be 2-dimensional, given input with different dimension or empty",
    ):
        cov_ssi_algorithm.apply(signal=input_signals, sampling_frequency=60)


@pytest.mark.parametrize("optimized", [True, False])
def test_cov_ssi_should_raise_error_if_impulse_response_computation_fails(
    optimized, mocker: MockerFixture
):
    algorithm = CovSSI(frequency_max=5, frequency_min=0, number_of_fft_points=2**4)
    signal = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])

    mocker.patch.object(CovSSI, "_compute_impulse_response", side_effect=ValueError("test"))
    mocker.patch.object(CovSSI, "_compute_impulse_response_optimized", side_effect=ValueError("test"))
    with pytest.raises(
        ModalIdentificationError,
        match="Impulse response computation failed: test",
    ):
        algorithm.apply(signal=signal, sampling_frequency=60, optimized=optimized)

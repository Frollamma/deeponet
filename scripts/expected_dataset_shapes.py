def operator_dims(Nx, M, trunk_dim=1):
    branch_dim = Nx * M
    return branch_dim, trunk_dim


def report_shapes():
    configs = [
        # -------------------------
        # operator learning problems
        # -------------------------
        {
            "id": "legendre_transform",
            "name": "Legendre transform",
            "problem_type": "operator",
            "npoints_output": 20,
            "num_train": 10000,
            "num_test": 100000,
            "Nx": 100,
            "M": 1,
            "trunk_dim": 1,
            "expand": True,
        },
        {
            "id": "ode_antiderivative",
            "name": "ODE Antiderivative",
            "problem_type": "ode",
            "num_train": 10000,
            "num_test": 100000,
            "branch_dim": 100,
            "trunk_dim": 1,
            "expand": False,
        },
        {
            "id": "ode_nonlinear",
            "name": "ODE Nonlinear",
            "problem_type": "ode",
            "num_train": 10000,
            "num_test": 100000,
            "branch_dim": 100,
            "trunk_dim": 1,
            "expand": False,
        },
        {
            "id": "ode_pendulum",
            "name": "ODE Pendulum",
            "problem_type": "ode",
            "num_train": 10000,
            "num_test": 100000,
            "branch_dim": 100,
            "trunk_dim": 1,
            "expand": False,
        },
        {
            "id": "diffusion_reaction",
            "name": "Diffusion Reaction",
            "problem_type": "pde",
            "npoints_output": 100,
            "num_train": 10000,
            "num_test": 100000,
            "Nx": 100,
            "M": 1,
            "trunk_dim": 2,
            "expand": True,
        },
        {
            "id": "convection",
            "name": "Convection",
            "problem_type": "pde",
            "npoints_output": 100,
            "num_train": 10000,
            "num_test": 100000,
            "Nx": 100,
            "M": 1,
            "trunk_dim": 2,
            "expand": True,
        },
        {
            "id": "advection_diffusion",
            "name": "Advection Diffusion",
            "problem_type": "pde",
            "npoints_output": 100,
            "num_train": 10000,
            "num_test": 100000,
            "Nx": 100,
            "M": 1,
            "trunk_dim": 2,
            "expand": True,
        },
        {
            "id": "stochastic_pde",
            "name": "Stochastic PDE",
            "problem_type": "pde",
            "npoints_output": 100,
            "num_train": 1000,
            "num_test": 1000,
            "Nx": 30,
            "M": 8,
            "trunk_dim": 1,
            "expand": True,
        },
        # -------------------------
        # fractional / structured problems
        # -------------------------
        {
            "id": "caputo_1d",
            "name": "Caputo 1D",
            "problem_type": "fractional_pde",
            "Num_u_train": 10000,
            "Num_u_test": 10000,
            "Num_x": 15,
            "Num_y": 10,
            "Num_alpha": 10,
            "branch_dim": 15,
            "trunk_dim": 2,
        },
        {
            "id": "fractional_laplacian_2d",
            "name": "2D Fractional Laplacian",
            "problem_type": "fractional_pde",
            "Num_u_train": 5000,
            "Num_u_test": 5000,
            "Num_x": 15,
            "Num_y": 15,
            "Num_alpha": 10,
            "branch_dim": 15 * 15,
            "trunk_dim": 3,
        },
    ]

    print("\n=== Dataset shape report ===\n")

    for config in configs:

        if config["problem_type"] == "ode":
            num_samples = config["num_train"] + config["num_test"]
            branch_dim = config["branch_dim"]
            trunk_dim = config["trunk_dim"]

        elif config["problem_type"] in ["operator", "pde"]:
            num_samples = (config["num_train"] + config["num_test"]) * config[
                "npoints_output"
            ]
            branch_dim, trunk_dim = operator_dims(
                Nx=config["Nx"],
                M=config["M"],
                trunk_dim=config["trunk_dim"],
            )

        elif config["problem_type"] == "fractional_pde":

            if config["id"] == "caputo_1d":
                per_function = config["Num_y"] * config["Num_alpha"]
                num_samples = (
                    config["Num_u_train"] + config["Num_u_test"]
                ) * per_function

            elif config["id"] == "fractional_laplacian_2d":
                per_function = (config["Num_y"] ** 2) * config["Num_alpha"]
                num_samples = (
                    config["Num_u_train"] + config["Num_u_test"]
                ) * per_function

            else:
                raise ValueError("unknown fractional_pde config")

            branch_dim = config["branch_dim"]
            trunk_dim = config["trunk_dim"]

        else:
            raise ValueError("unknown problem_type")

        print(f"problem: {config['name']}")
        print(f"num_samples: {num_samples}")
        print(f"branch_input_dim: {branch_dim}")
        print(f"trunk_input_dim: {trunk_dim}")
        print()


if __name__ == "__main__":
    report_shapes()

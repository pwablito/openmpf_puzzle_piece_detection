# syntax=docker/dockerfile:experimental

ARG BUILD_REGISTRY=openmpf/
ARG BUILD_TAG=9.0.0
FROM ${BUILD_REGISTRY}openmpf_python_executor_ssb:${BUILD_TAG}

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y libgl1-mesa-glx

RUN pip3 install --upgrade pip

RUN pip3 install ultralytics

ARG RUN_TESTS=false

RUN --mount=target=.,readwrite \
    cp -r models / ; \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_puzzle_piece_detection.py; fi

LABEL org.label-schema.name="OpenMPF Puzzle Piece Detection"
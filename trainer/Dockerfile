FROM rust:latest
WORKDIR /app

RUN mkdir codebase
RUN mkdir final

# Cmake dependency
RUN apt-get update
RUN apt-get -y install cmake

# Perf dependency
RUN apt-get -y install linux-perf
RUN cp /usr/bin/perf_5.10 /usr/bin/perf_5.10.60

# Flamegraph dependency
RUN cargo install flamegraph

COPY ./temp/chess/config.xml /app/files/chess/
COPY ./temp/chess/endgames.dat /app/files/chess/
COPY ./temp/chess/openings.dat /app/files/chess/

WORKDIR /app/codebase
COPY ./codebase/Cargo.toml .
# Removes benchmarks from dependencies
RUN sed -i 's|\[\[bench.*||g' Cargo.toml &&  \
    sed -i 's|name = "conv_layer"||g' Cargo.toml &&  \
    sed -i 's|harness = false||g' Cargo.toml

# Download and compile codebase dependencies
RUN mkdir src && echo "// dummy file" > src/lib.rs && cargo build --release && rm src/lib.rs

WORKDIR /app/final
COPY trainer/Cargo.toml .
# Removes codebase from dependencies
RUN sed -i 's|codebase = .*||g' Cargo.toml
# Download and compile server dependencies
RUN mkdir src && echo "fn main(){}" > src/main.rs && cargo build --release && rm src/main.rs

# Copy the rest of the files
WORKDIR /app/codebase
COPY ./codebase/src ./src

WORKDIR /app/final
COPY trainer/Cargo.toml .
COPY trainer/src ./src
RUN cargo build --release

ENV MOUNTED_PATH="/app/files"
ENV NAME="chess"
ENV VERSIONS="1"
ENV VERSIONS_SERVER_URL="https://ai-playground-server.livelybay-b5b6ca38.brazilsouth.azurecontainerapps.io"
ENV EPOCHS_PER_VERSION="2"

ENTRYPOINT ["tail", "-f", "/dev/null"]

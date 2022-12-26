FROM rust:latest as builder
WORKDIR /app

RUN mkdir codebase
RUN mkdir final

# Cmake dependency
RUN apt-get update
RUN apt-get -y install cmake

#ENTRYPOINT ["tail", "-f", "/dev/null"]

WORKDIR /app/codebase
COPY ./codebase/Cargo.toml .
# Removes benchmarks from dependencies
RUN sed -i 's|\[\[bench.*||g' Cargo.toml &&  \
    sed -i 's|name = "conv_layer"||g' Cargo.toml &&  \
    sed -i 's|harness = false||g' Cargo.toml

# Download and compile codebase dependencies
RUN mkdir src && echo "// dummy file" > src/lib.rs && cargo build --release && rm src/lib.rs

WORKDIR /app/final
COPY versions_server/Cargo.toml .
# Removes codebase from dependencies
RUN sed -i 's|codebase = .*||g' Cargo.toml
# Download and compile server dependencies
RUN mkdir src && echo "fn main(){}" > src/main.rs && cargo build --release && rm src/main.rs

# Copy the rest of the files
WORKDIR /app/codebase
COPY ./codebase/src ./src

WORKDIR /app/final
COPY versions_server/Cargo.toml .
COPY versions_server/src ./src
RUN cargo build --release

FROM debian:bullseye-slim as runner
WORKDIR /app
RUN mkdir build
COPY --from=builder /app/final/target/release /app/build
WORKDIR /app/build

ENV MOUNTED_PATH="/app/files"
ENV HOST_ADDRESS="0.0.0.0"
ENV PORT="443"

RUN mkdir /app/files
RUN chmod -R 777 /app/files
EXPOSE 443
CMD ["./versions_server"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
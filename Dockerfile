# TODO: production mode
FROM rust:latest as builder
WORKDIR /app

RUN mkdir codebase
RUN mkdir server

WORKDIR /app/codebase
COPY codebase/Cargo.toml .
# Download and compile codebase dependencies
RUN mkdir src && echo "// dummy file" > src/lib.rs && cargo build && rm src/lib.rs

WORKDIR /app/server
COPY server/Cargo.toml .
# Removes codebase from dependencies
RUN sed -i 's|codebase = .*||g' Cargo.toml
# Download and compile server dependencies
RUN mkdir src && echo "fn main(){}" > src/main.rs && cargo build && rm src/main.rs

# Copy the rest of the files
WORKDIR /app/codebase
COPY codebase/build.rs .
COPY codebase/src ./src

WORKDIR /app/server
COPY server/Cargo.toml .
COPY server/src ./src
RUN cargo build

FROM debian:bullseye-slim as runner
WORKDIR /app
RUN mkdir build
COPY --from=builder /app/server/target/debug /app/build
WORKDIR /app/build

ENV ROCKET_ADDRESS=0.0.0.0
ENV MODEL_FILE_PATH="/app/files"

EXPOSE 8000
RUN mkdir /app/files
RUN chmod -R 777 /app/files
COPY config/digits /app/files/digits
CMD ["./server"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
FROM rust:latest as builder
WORKDIR /app

RUN mkdir codebase
RUN mkdir server

WORKDIR /app/codebase
COPY codebase/Cargo.lock .
COPY codebase/Cargo.toml .

# Download and compile codebase dependencies
RUN mkdir src && echo "// dummy file" > src/lib.rs && cargo build --target x86_64-unknown-linux-gnu && rm src/lib.rs

COPY codebase/build.rs .
COPY codebase/src ./src

WORKDIR /app/server
COPY server/Cargo.lock .
COPY server/Cargo.toml .

# Download and compile server dependencies
RUN mkdir src && echo "fn main(){}" > src/main.rs && cargo build --target x86_64-unknown-linux-gnu && rm src/main.rs

COPY server/src ./src
RUN cargo build

FROM debian:bullseye-slim as runner
WORKDIR /app
RUN mkdir build
COPY --from=builder /app/server/target/debug /app/build
WORKDIR /app/build

ENV ROCKET_ADDRESS=0.0.0.0
EXPOSE 8000
RUN mkdir /files
RUN chmod -R 777 /files
CMD ["./server"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
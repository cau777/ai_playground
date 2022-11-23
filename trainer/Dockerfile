FROM rust:latest as builder
WORKDIR /app

RUN mkdir codebase
RUN mkdir final

WORKDIR /app/codebase
COPY ./codebase/Cargo.toml .
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

FROM debian:bullseye-slim as runner
WORKDIR /app
RUN mkdir build
COPY --from=builder /app/final/target/release /app/build
WORKDIR /app/build

ENV MOUNTED_PATH="/app/files"
RUN mkdir /app/files
RUN chmod -R 777 /app/files
CMD ["./trainer"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        back: {
          0: "#212529",
          1: "#343a40",
          2: "#495057",
          3: "#6c757d"
        },
        font: {
          0: "#f8f9fa",
          1: "#e9ecef",
          2: "#dee2e6",
          3: "#ced4da"
        },
        primary: {
          100: "#ffba08",
          200: "#faa307",
          300: "#f48c06",
          400: "#e85d04",
          500: "#dc2f02",
          600: "#d00000",
          700: "#9d0208",
          800: "#6a040f",
          900: "#370617",
        },
      }
    },
  },
  plugins: [],
}

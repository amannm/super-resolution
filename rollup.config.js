import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";

export default {
  input: "./dist/main.js",
  plugins: [
    resolve(),
    commonjs({
      ignore: ["crypto"]
    })
  ],
  output: {
    file: "./web/app.js",
    format: "es"
  }
};
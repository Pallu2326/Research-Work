{
  description = "Python Dev Environment for ML";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.python312
          pkgs.uv
          pkgs.python312Packages.numpy
          pkgs.python312Packages.matplotlib
          pkgs.stdenv.cc.cc.lib
        ];

        shellHook = ''
          echo "Python $(python --version) Dev Environment for ML"
        '';
      };
    };
}

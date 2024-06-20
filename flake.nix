{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self
  , nixpkgs
  , flake-utils
  }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShells = {
          default = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.python3.pkgs.python-lsp-server
              pkgs.python3.pkgs.numpy
              pkgs.python3.pkgs.scipy
              pkgs.python3.pkgs.matplotlib
              pkgs.python3.pkgs.pygame
            ];
          };
        };
      }
    );
}

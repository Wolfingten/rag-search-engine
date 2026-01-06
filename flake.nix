{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];

    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    devShells = forAllSystems (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            python312
          ];
          shellHook = ''
            # Initialize uv project if needed
            if [ ! -f "pyproject.toml" ]; then
              echo "Initializing uv project..."
              uv init
            fi

            # Create python virtual env if needed
            if [ ! -d ".venv" ]; then
              echo "Creating Python virtual environment with uv..."
              uv venv
            fi

            # activate python venv automatically
            source .venv/bin/activate
          '';
        };
      }
    );
  };
}

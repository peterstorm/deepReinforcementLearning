let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs {};
  unstable = import sources.unstable {};
  self = {
    inherit pkgs;
    inherit unstable;
  };

in
  self

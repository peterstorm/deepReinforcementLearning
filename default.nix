{ src ? import ./nixpkgs.nix }:

let
  pkgs = src.pkgs;
  inherit (pkgs) stdenv lib;
  unstable = src.unstable;
  pythonPackages = pkgs.python37Packages;
  pythonEnv = with pythonPackages; 
        [ 
          setuptools
          unstable.pkgs.python37Packages.pip
          pylint
          jedi
          virtualenv
          pyopengl
        ];
  name = "grokkingRDL";

in 
  {
    shell = pkgs.mkShell {
      buildInputs = 
        [ 
          pythonEnv
          pkgs.glib
          pkgs.glibc
          pkgs.freeglut
          pkgs.libGLU
          pkgs.xorg.libX11
        ];
      shellHook = ''
        echo 'Entering Python Project Environment'
        # extra packages can be installed here
         export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/:${pkgs.glib.out}/lib/:${pkgs.freeglut.out}/lib/:${pkgs.libGLU.out}/lib/:${pkgs.xorg.libX11.out}/lib/
        unset SOURCE_DATE_EPOCH
      '';
      };
    }

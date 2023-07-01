{ sources ? import ./nix/sources.nix
, pkgs ? import sources.nixpkgs {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  }
}:

pkgs.mkShell {
  packages = with pkgs; [
    (python3.withPackages (ps: [
      ps.pytorch-bin
    ]))

    which
    htop
    neofetch
  ];

  shellHook = ''
    neofetch
  '';

  MY_ENVIRONMENT_VARIABLE = "world";
}

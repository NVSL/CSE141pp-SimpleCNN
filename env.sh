export CANELA_ROOT=$PWD

if [ "$ARCHLAB_ROOT." = "." ]; then
    pushd cse141pp-archlab/ #use our local copy for it's build system
    . env.sh
    popd
fi

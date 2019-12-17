if [ "$ARCHLAB_ROOT." = "." ]; then
    pushd cse141pp-archlab/;
    . env.sh
    popd
fi

export CANELA_ROOT=$PWD

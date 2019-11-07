if [ "$ARCHLAB_ROOT." = "." ]; then
    pushd cse141pp-archlab/;
    . env.sh
    popd
fi

if [ "$GOOGLE_TEST_ROOT." = "." ]; then
    export GOOGLE_TEST_ROOT=$PWD/googletest
fi

if [ "$CANELA_ROOT." = "." ]; then
    export CANELA_ROOT=$PWD
fi

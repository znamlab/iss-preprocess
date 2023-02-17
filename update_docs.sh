#
SRCDOCS=$PWD/docs/build/html

cd $PWD/docs
make clean
make html

cd $SRCDOCS
MSG="Updating docs"

TMPREPO=$PWD/.tmp
rm -rf $TMPREPO
mkdir -p -m 0755 $TMPREPO

git clone git@github.com:znamlab/iss-preprocess.git $TMPREPO
cd $TMPREPO
git checkout gh-pages  ###gh-pages has previously one off been set to be nothing but html
cp -r $SRCDOCS/ $TMPREPO
git add -A
git commit -m "$MSG" && git push origin gh-pages
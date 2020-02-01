
/*
文字列で構成するpathの配列をjsonのtree構造に変換するプログラム
*/
var _ = require('lodash');
var paths = [
    '/FolderC/FolderA/FolderQ/ItemA',
    '/FolderC/FolderA/Item1',
    '/FolderD/FolderF/FolderM/ItemA',
    '/FolderD/FolderF/FolderM/ItemB',
    '/FolderD/FolderG/ItemD',
    '/ItemInRoot'
];

function pathString2Tree(paths, cb) {
    var tree = [];

    //ループする！
    _.each(paths, function (path) {
        // currentLevelを rootに初期化する
        var pathParts = path.split('/');
        pathParts.shift();
        // currentLevelを rootに初期化する
        var currentLevel = tree;

        _.each(pathParts, function (part) {

            // pathが既存しているかどうかをチェックする
            var existingPath = _.find(currentLevel, {
                name: part
            });

            if (existingPath) {
                // Pathはすでにツリー構造に入っているので、追加しない
                // current levelを下の子供階層に設定し直す
                currentLevel = existingPath.children;
            } else {
                var newPart = {
                    name: part,
                    children: [],
                }

                currentLevel.push(newPart);
                currentLevel = newPart.children;
            }
        });
    });

    cb(tree);
}
pathString2Tree(paths, function (tree) {
    console.log('tree: ', JSON.stringify(tree));
});
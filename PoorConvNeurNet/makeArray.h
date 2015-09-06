//
//  makeArray.h
//  CharacterRecognition
//
//  Created by IGUCHI Yusuke on 2015/08/25.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#ifndef CharacterRecognition_makeArray_h
#define CharacterRecognition_makeArray_h
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include "Definitions.h"
#define CLASSES 240 //文字ファイルの数(ここでは、小学1,2年の漢字のみを扱うため)

//正規化後の画像のサイズとフォントを指定すると返してくれる。
void GetDataMat(int width, std::string font, std::vector<MatrixF> DatasArray){
    std::string basePath("/Users/iguchiyusuke/cpp/CharacterRecognition/CharacterRecognition/testdata/"+std::to_string(width)+"/"+font+"/");
    std::vector<std::string> pathList(CLASSES,basePath);
    //各文字の正規化後の画像のデータは.txtで保存されている。
    char filenames[CLASSES][10] = {{"19968.txt"},{"21491.txt"},{"38632.txt"},{"20870.txt"},{"29579.txt"},{"38899.txt"},
        {"19979.txt"},{"28779.txt"},{"33457.txt"},{"35997.txt"},{"23398.txt"},{"27671.txt"},
        {"20061.txt"},{"20241.txt"},{"29577.txt"},{"37329.txt"},{"31354.txt"},{"26376.txt"},
        {"29356.txt"},{"35211.txt"},{"20116.txt"},{"21475.txt"},{"26657.txt"},{"24038.txt"},
        {"19977.txt"},{"23665.txt"},{"23376.txt"},{"22235.txt"},{"31992.txt"},{"23383.txt"},
        {"32819.txt"},{"19971.txt"},{"36554.txt"},{"25163.txt"},{"21313.txt"},{"20986.txt"},
        {"22899.txt"},{"23567.txt"},{"19978.txt"},{"26862.txt"},{"20154.txt"},{"27700.txt"},
        {"27491.txt"},{"29983.txt"},{"38738.txt"},{"22805.txt"},{"30707.txt"},{"36196.txt"},
        {"21315.txt"},{"24029.txt"},{"20808.txt"},{"26089.txt"},{"33609.txt"},{"36275.txt"},
        {"26449.txt"},{"22823.txt"},{"30007.txt"},{"31481.txt"},{"20013.txt"},{"34411.txt"},
        {"30010.txt"},{"22825.txt"},{"30000.txt"},{"22303.txt"},{"20108.txt"},{"26085.txt"},
        {"20837.txt"},{"24180.txt"},{"30333.txt"},{"20843.txt"},{"30334.txt"},{"25991.txt"},
        {"26408.txt"},{"26412.txt"},{"21517.txt"},{"30446.txt"},{"31435.txt"},{"21147.txt"},
        {"26519.txt"},{"20845.txt"},{"24341.txt"},{"32701.txt"},{"38642.txt"},{"22290.txt"},
        {"36960.txt"},{"20309.txt"},{"31185.txt"},{"22799.txt"},{"23478.txt"},{"27468.txt"},
        {"30011.txt"},{"22238.txt"},{"20250.txt"},{"28023.txt"},{"32117.txt"},{"22806.txt"},
        {"35282.txt"},{"27005.txt"},{"27963.txt"},{"38291.txt"},{"20024.txt"},{"23721.txt"},
        {"38996.txt"},{"27773.txt"},{"35352.txt"},{"24112.txt"},{"24339.txt"},{"29275.txt"},
        {"39770.txt"},{"20140.txt"},{"24375.txt"},{"25945.txt"},{"36817.txt"},{"20804.txt"},
        {"24418.txt"},{"35336.txt"},{"20803.txt"},{"35328.txt"},{"21407.txt"},{"25144.txt"},
        {"21476.txt"},{"21320.txt"},{"24460.txt"},{"35486.txt"},{"24037.txt"},{"20844.txt"},
        {"24195.txt"},{"20132.txt"},{"20809.txt"},{"32771.txt"},{"34892.txt"},{"39640.txt"},
        {"40644.txt"},{"21512.txt"},{"35895.txt"},{"22269.txt"},{"40658.txt"},{"20170.txt"},
        {"25165.txt"},{"32048.txt"},{"20316.txt"},{"31639.txt"},{"27490.txt"},{"24066.txt"},
        {"30690.txt"},{"22985.txt"},{"24605.txt"},{"32025.txt"},{"23546.txt"},{"33258.txt"},
        {"26178.txt"},{"23460.txt"},{"31038.txt"},{"24369.txt"},{"39318.txt"},{"31179.txt"},
        {"36913.txt"},{"26149.txt"},{"26360.txt"},{"23569.txt"},{"22580.txt"},{"33394.txt"},
        {"39135.txt"},{"24515.txt"},{"26032.txt"},{"35242.txt"},{"22259.txt"},{"25968.txt"},
        {"35199.txt"},{"22768.txt"},{"26143.txt"},{"26228.txt"},{"20999.txt"},{"38634.txt"},
        {"33337.txt"},{"32218.txt"},{"21069.txt"},{"32068.txt"},{"36208.txt"},{"22810.txt"},
        {"22826.txt"},{"20307.txt"},{"21488.txt"},{"22320.txt"},{"27744.txt"},{"30693.txt"},
        {"33590.txt"},{"26172.txt"},{"38263.txt"},{"40165.txt"},{"26397.txt"},{"30452.txt"},
        {"36890.txt"},{"24351.txt"},{"24215.txt"},{"28857.txt"},{"38651.txt"},{"20992.txt"},
        {"20908.txt"},{"24403.txt"},{"26481.txt"},{"31572.txt"},{"38957.txt"},{"21516.txt"},
        {"36947.txt"},{"35501.txt"},{"20869.txt"},{"21335.txt"},{"32905.txt"},{"39340.txt"},
        {"22770.txt"},{"36023.txt"},{"40614.txt"},{"21322.txt"},{"30058.txt"},{"29238.txt"},
        {"39080.txt"},{"20998.txt"},{"32862.txt"},{"31859.txt"},{"27497.txt"},{"27597.txt"},
        {"26041.txt"},{"21271.txt"},{"27598.txt"},{"22969.txt"},{"19975.txt"},{"26126.txt"},
        {"40180.txt"},{"27611.txt"},{"38272.txt"},{"22812.txt"},{"37326.txt"},{"21451.txt"},
        {"29992.txt"},{"26332.txt"},{"26469.txt"},{"37324.txt"},{"29702.txt"},{"35441.txt"}};
    
    //各文字についてテキストファイルからのデータの読み込みと行列への格納を行う
    for(int f = 0; f < CLASSES; f++){
        pathList[f] += std::string(filenames[f]);
        std::ifstream ifs(pathList[f]);
        std::cout<< f << ":succeed in opening : " << filenames[f]<<std::endl;
        if(ifs.fail()){
            std::cerr<< "サンプルデータ読み込み失敗\nGetDataMatの引数が間違っていませんか?\nプログラムを終了します。" << std::endl;
            ifs.close();
            exit(1);
        }
        std::istreambuf_iterator<char> it(ifs);
        std::istreambuf_iterator<char> last;
        std::string datas_str(it,last);
        for(int j = 0; j < DatasArray.at(f).rows; ++j){
            for(int i = 0; i < DatasArray.at(f).cols; ++i){
                //'0'の文字コードは48
                DatasArray.at(f).at<float>(j,i) = (datas_str[j*width+i] == 48)? 0 : 1;
            }
        }
        ifs.close();
    }
}

#endif

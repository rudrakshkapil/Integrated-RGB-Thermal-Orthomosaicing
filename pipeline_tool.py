import os
from PyQt5 import QtWidgets, QtCore, QtGui
from GUI_generated import Ui_MainWindow
import sys
import pprint
import yaml
import webbrowser


from PyQt5.QtGui import QIntValidator, QDoubleValidator




class Stream(QtCore.QObject):
    newText = QtCore.pyqtSignal(str)
    def write(self, text):
        self.newText.emit(str(text))



class PipelineTool(Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        # buttons
        self.buttonRestoreSettings.clicked.connect(self.restoreDefaultSettings)
        self.buttonSaveSettings.clicked.connect(self.saveButtonClicked)
        self.buttonStartProcessing.clicked.connect(self.startProcessingButtonClicked)

        # file explorer buttons
        self.buttonProjectDir.clicked.connect(lambda: self.fileExplorerClicked(self.lineProjectDir))
        self.buttonMappingDir.clicked.connect(lambda: self.fileExplorerClicked(self.lineMappingDir))
        self.buttonRGBDir.clicked.connect(lambda: self.fileExplorerClicked(self.lineRGBDir))
        self.buttonThermalDir.clicked.connect(lambda: self.fileExplorerClicked(self.lineThermalDir))

        # running pipeline process
        self.pipeline_process = QtCore.QProcess(self)
        # QProcess emits `readyRead` when there is data to be read
        # self.pipeline_process.readyRead.connect(self.dataReady)
        self.pipeline_process.started.connect(lambda: self.buttonStartProcessing.setEnabled(False))
        self.pipeline_process.finished.connect(lambda: self.buttonStartProcessing.setEnabled(True))

        # menu bar links
        self.actionPipeline.triggered.connect(lambda: self.openLink('pipeline'))
        self.actionODM.triggered.connect(lambda: self.openLink('ODM'))
        self.actionTree_Detection.triggered.connect(lambda: self.openLink('Deepforest'))

        # validators
        self.onlyInt = QIntValidator()
        self.onlyDouble = QDoubleValidator()
        

        # apply validators to line edits
        self.lineTDScoreThresh.setValidator(self.onlyDouble)
        self.lineTDPatchOverlap.setValidator(self.onlyDouble)
        self.lineTDIOUThresh.setValidator(self.onlyDouble)
        self.lineTDNumCrowns.setValidator(self.onlyInt)
        self.lineTDPatchSize.setValidator(self.onlyInt)

        self.lineRGBCropScale.setValidator(self.onlyDouble)

        self.lineNGFDownscale.setValidator(self.onlyDouble)
        self.lineNGFLearningRate.setValidator(self.onlyDouble)
        # self.lineNGFMILowerBound.setValidator(self.onlyDouble)
        self.lineNGFBatchSize.setValidator(self.onlyInt)
        self.lineNGFLevels.setValidator(self.onlyInt)
        self.lineNGFNumIters.setValidator(self.onlyInt)
        self.lineManualImagePairIndex.setValidator(self.onlyInt)

        self.l_auto_boundary_distance.setValidator(self.onlyDouble)
        self.l_crop.setValidator(self.onlyDouble)
        self.l_dem_resolution.setValidator(self.onlyDouble)
        self.l_depthmap_resolution.setValidator(self.onlyDouble)
        self.l_gps_accuracy.setValidator(self.onlyDouble)
        self.l_orthophoto_resolution.setValidator(self.onlyDouble)
        self.l_pc_filter.setValidator(self.onlyDouble)
        self.l_pc_sample.setValidator(self.onlyDouble)
        self.l_smrf_scalar.setValidator(self.onlyDouble)
        self.l_smrf_slope.setValidator(self.onlyDouble)
        self.l_smrf_threshold.setValidator(self.onlyDouble)
        self.l_smrf_window.setValidator(self.onlyDouble)

        self.l_dem_decimation.setValidator(self.onlyInt)
        self.l_dem_gapfill_steps.setValidator(self.onlyInt)
        self.l_matcher_neighbors.setValidator(self.onlyInt)
        self.l_max_concurrency.setValidator(self.onlyInt)
        self.l_mesh_octree_depth.setValidator(self.onlyInt)
        self.l_mesh_size.setValidator(self.onlyInt)
        self.l_min_num_features.setValidator(self.onlyInt)
        self.l_resize_to.setValidator(self.onlyInt)
        self.l_rolling_shutter_readout.setValidator(self.onlyInt)
        self.l_split.setValidator(self.onlyInt)
        self.l_split_overlap.setValidator(self.onlyInt)

        self.lineDJIEmissivity.setValidator(self.onlyDouble)
        self.lineDJIHumidity.setValidator(self.onlyDouble)
        self.lineDJIDistance.setValidator(self.onlyDouble)
        self.lineDJITemperature.setValidator(self.onlyDouble)

    # ---------------- For output stream of this file
        sys.stdout = Stream(newText=self.onUpdateText)
        sys.stderr = Stream(newText=self.onUpdateText)

        # load config:
        try:
            with open('configs/combined.yml', 'rb') as f:
                cfg = yaml.safe_load(f.read())
            self.restoreSettingsfromCfg(cfg)
        except:
            self.restoreDefaultSettings()

    def onUpdateText(self, text):
        cursor = self.outputText.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.outputText.setTextCursor(cursor)
        self.outputText.ensureCursorVisible()

    def __del__(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stdout__
    # --------------

    def openLink(self, option):
        URL_dict = {'pipeline': 'https://integrated-rgb-thermal-orthomosaicing.readthedocs.io/en/latest/',
                    'ODM': 'https://docs.opendronemap.org/',
                    'Deepforest':'https://deepforest.readthedocs.io/en/latest/'}
        webbrowser.open(URL_dict[option])

    def saveGeneralSettings(self, cfg):
        ''' General settings '''
        # lists: (mode, camera)
        cfg['STAGES']['METHOD'] = self.listMethod.currentItem().text()
        cfg['CAMERA'] = self.listCamera.currentItem().text()

        # checkboxes:
        cfg['STAGES']['THERMAL']['TWO_STEP'] = self.checkThermalTwoStep.isChecked()
        cfg['OUTPUT']['DENORMALIZE'] =  self.checkDenormalizeOrtho.isChecked()
        cfg['OUTPUT']['OPTIMIZE_DISK_SPACE'] = self.checkOptimizeDiskSpace.isChecked()

        # checkboxes: (stages)
        cfg['STAGES']['COMBINED_STAGES']['STAGE_1'] = self.checkStage1.isChecked()
        cfg['STAGES']['COMBINED_STAGES']['STAGE_2'] = self.checkStage2.isChecked()
        cfg['STAGES']['COMBINED_STAGES']['STAGE_3'] = self.checkStage3.isChecked()
        cfg['STAGES']['COMBINED_STAGES']['STAGE_4'] = self.checkStage4.isChecked()

        # TODO: disable stuff accordingly
        # if cfg['STAGES']['METHOD'] == 'combined':


        return cfg

    def saveDownstreamSettings(self, cfg):
        ''' Downstream Task Settings'''
        # checkboxes
        cfg['STAGES']['TREE_DETECTION']['PERFORM'] = self.checkTDPerform.isChecked()
        cfg['STAGES']['TREE_DETECTION']['SAVE_BBOX_IMAGE'] = self.checkTDSaveBBoxOrtho.isChecked()
        cfg['STAGES']['TREE_DETECTION']['SAVE_TREE_CROWNS'] = self.checkTDSaveTreeCrowns.isChecked()

        # lines: 
        cfg['STAGES']['TREE_DETECTION']['SCORE_THRESH'] = float(self.lineTDScoreThresh.text())
        cfg['STAGES']['TREE_DETECTION']['PATCH_OVERLAP'] = float(self.lineTDPatchOverlap.text())
        cfg['STAGES']['TREE_DETECTION']['IOU_THRESH'] = float(self.lineTDIOUThresh.text())
        cfg['STAGES']['TREE_DETECTION']['PATCH_SIZE'] = int(self.lineTDPatchSize.text())
        cfg['STAGES']['TREE_DETECTION']['N_TOP'] = int(self.lineTDNumCrowns.text())

        return cfg

    def savePreprocessingSettings(self, cfg):
        ''' Preprocessing settings (RGB & Thermal) '''
        # RGB
        cfg['PREPROCESSING']['RGB']['CROP'] = self.checkRGBCrop.isChecked()
        cfg['PREPROCESSING']['RGB']['SCALE'] = float(self.lineRGBCropScale.text())

        # Thermal
        cfg['PREPROCESSING']['THERMAL']['NORMALIZE'] = self.checkNormalize.isChecked()
        cfg['PREPROCESSING']['THERMAL']['H20T']['FUNCTION'] = self.listDJIFunction.currentItem().text()
        cfg['PREPROCESSING']['THERMAL']['H20T']['EMISSIVITY'] = float(self.lineDJIEmissivity.text())
        cfg['PREPROCESSING']['THERMAL']['H20T']['TEMPERATURE'] = float(self.lineDJITemperature.text())
        cfg['PREPROCESSING']['THERMAL']['H20T']['DISTANCE'] = float(self.lineDJIDistance.text())
        cfg['PREPROCESSING']['THERMAL']['H20T']['HUMIDITY'] = float(self.lineDJIHumidity.text())
        return cfg

    def saveHomographySettings(self, cfg):
        ''' Homography settings for stage 3 '''
        # mode
        cfg['HOMOGRAPHY']['MODE'] = self.listHomographyMode.currentItem().text().upper()
        cfg['HOMOGRAPHY']['DOF'] = self.listDOF.currentItem().text()

        # NGF
        cfg['HOMOGRAPHY']['NGF_PARAMS']['BATCH_SIZE'] = int(self.lineNGFBatchSize.text())
        cfg['HOMOGRAPHY']['NGF_PARAMS']['HIST_EQ'] = self.checkNGFHistEq.isChecked()
        cfg['HOMOGRAPHY']['NGF_PARAMS']['LEVELS'] = int(self.lineNGFLevels.text())
        cfg['HOMOGRAPHY']['NGF_PARAMS']['DOWNSCALE'] = float(self.lineNGFDownscale.text())
        cfg['HOMOGRAPHY']['NGF_PARAMS']['LEARNING_RATE'] = float(self.lineNGFLearningRate.text())
        cfg['HOMOGRAPHY']['NGF_PARAMS']['N_ITERS'] = int(self.lineNGFNumIters.text())
        # cfg['HOMOGRAPHY']['NGF_PARAMS']['MI_LOWER_BOUND'] = float(self.lineNGFMILowerBound.text())
        cfg['HOMOGRAPHY']['NGF_PARAMS']['TWO_WAY_LOSS'] = self.checkNGFTwoWayLoss.isChecked()
        cfg['HOMOGRAPHY']['NGF_PARAMS']['SYSTEMATIC_SAMPLE'] = self.checkNGFSystematic.isChecked()
        
        # manual
        cfg['HOMOGRAPHY']['MANUAL_PARAMS']['IMAGE_PAIR_INDEX'] = int(self.lineManualImagePairIndex.text())
        cfg['HOMOGRAPHY']['MANUAL_PARAMS']['HIST_EQ'] = self.checkManualHistEq.isChecked()

        return cfg

    def fileExplorerClicked(self, lineEdit):
        ''' Navigate to directory and store in corresponding line edit '''
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Please select a directory', '.')
        lineEdit.setText(dir_path)
        print("Selected", dir_path)

    def saveInputDirectorySettings(self, cfg):
        ''' Save paths '''
        cfg['DIR']['PROJECT'] = self.lineProjectDir.text()
        cfg['DIR']['H20T_DATA'] = self.lineMappingDir.text()
        cfg['DIR']['RGB'] = self.lineRGBDir.text()
        cfg['DIR']['THERMAL'] = self.lineThermalDir.text()

        return cfg


    def saveODMSettings(self, cfg):
        ''' Save relevant settings for ODM processing '''

        # get all param names
        with open('resources/ODM_params.yml', 'rb') as f:
            ODM_params = yaml.safe_load(f.read())

        # convert value representation to functions
        func_map = {'int': int, 'float':float, 'str':str}
        for key, val in ODM_params.items():
            if val is not None:
                ODM_params[key] = func_map[val]

        # loop over params
        ODM_dict = {}
        for param in ODM_params.keys():
            # get checkbox, determine toggled
            find_param = param.replace('-','_')
            cb = self.findChild(QtWidgets.QCheckBox, f"c_{find_param}")
            if cb is None:
                print(param, 'child is none')
                exit()
            if not cb.isChecked():
                continue

            # get line edit
            le = self.findChild(QtWidgets.QLineEdit, f"l_{find_param}")
            le = le.text() if le is not None else ''
            func = ODM_params[param]
            ODM_dict[param] = func(le) if func is not None else le
            
        # save to config and return
        cfg['ODM'] = ODM_dict
        return cfg
        

    def saveButtonClicked(self):
        '''
        Update config file
        '''
        # load config:
        with open('configs/combined.yml', 'rb') as f:
            cfg = yaml.safe_load(f.read())
        

        # # # settings
        # cfg = {}
        cfg = self.saveGeneralSettings(cfg)
        cfg = self.saveDownstreamSettings(cfg)
        cfg = self.savePreprocessingSettings(cfg)
        cfg = self.saveHomographySettings(cfg)
        cfg = self.saveInputDirectorySettings(cfg)
        cfg = self.saveODMSettings(cfg)


        # save config:
        with open('configs/combined.yml', 'w') as f:
            yaml.dump(cfg, f)
        pprint.pprint(cfg, width=1)
        print("Settings saved.")
        

    def restoreDefaultSettings(self):
        ''' Restore settings: save default.yml as combined.yml, and update GUI items'''
        # load config:
        with open('configs/default.yml', 'rb') as f:
            default_cfg = yaml.safe_load(f.read())

        self.restoreSettingsfromCfg(default_cfg)

        # save as combined
        with open('configs/combined.yml', 'w') as f:
            yaml.dump(default_cfg, f)
        pprint.pprint(default_cfg, width=1)
        print("Restored and saved default settings")
        
    def restoreSettingsfromCfg(self, cfg):
        ''' Restore settings:  and update GUI items from cfg'''
        # General
        self.listMethod.setCurrentRow(['combined','rgb_only','thermal_only'].index(cfg['STAGES']["METHOD"]))
        self.checkStage1.setChecked(cfg["STAGES"]["COMBINED_STAGES"]["STAGE_1"])
        self.checkStage2.setChecked(cfg["STAGES"]["COMBINED_STAGES"]["STAGE_2"])
        self.checkStage3.setChecked(cfg["STAGES"]["COMBINED_STAGES"]["STAGE_3"])
        self.checkStage4.setChecked(cfg["STAGES"]["COMBINED_STAGES"]["STAGE_4"])
        self.listCamera.setCurrentRow(['H20T','other'].index(cfg['CAMERA']))
        self.checkThermalTwoStep.setChecked(cfg["STAGES"]["THERMAL"]["TWO_STEP"])
        self.checkDenormalizeOrtho.setChecked(cfg["OUTPUT"]["DENORMALIZE"])
        self.checkOptimizeDiskSpace.setChecked(cfg["OUTPUT"]["OPTIMIZE_DISK_SPACE"])

        # input dirs - nothing
        self.lineProjectDir.setText(cfg["DIR"]["PROJECT"])

        # Downstream tasks
        self.checkTDPerform.setChecked(cfg['STAGES']['TREE_DETECTION']['PERFORM'])
        self.checkTDSaveBBoxOrtho.setChecked(cfg['STAGES']['TREE_DETECTION']['SAVE_BBOX_IMAGE'])
        self.checkTDSaveTreeCrowns.setChecked(cfg['STAGES']['TREE_DETECTION']['SAVE_TREE_CROWNS'])
        self.lineTDIOUThresh.setText(str(cfg['STAGES']['TREE_DETECTION']['IOU_THRESH'])) 
        self.lineTDScoreThresh.setText(str(cfg['STAGES']['TREE_DETECTION']['SCORE_THRESH']))
        self.lineTDPatchSize.setText(str(cfg['STAGES']['TREE_DETECTION']['PATCH_SIZE']))
        self.lineTDPatchOverlap.setText(str(cfg['STAGES']['TREE_DETECTION']['PATCH_OVERLAP']))
        self.lineTDNumCrowns.setText(str(cfg['STAGES']['TREE_DETECTION']['N_TOP']))

        # data preprocessing
        self.checkRGBCrop.setChecked(cfg['PREPROCESSING']['RGB']['CROP'])
        self.checkNormalize.setChecked(cfg['PREPROCESSING']['THERMAL']['NORMALIZE'])
        self.lineRGBCropScale.setText(str(cfg['PREPROCESSING']['RGB']['SCALE']))
        self.listDJIFunction.setCurrentRow(['extract','measure'].index(cfg['PREPROCESSING']['THERMAL']['H20T']['FUNCTION']))
        self.lineDJITemperature.setText(str(cfg['PREPROCESSING']['THERMAL']['H20T']['TEMPERATURE']))
        self.lineDJIHumidity.setText(str(cfg['PREPROCESSING']['THERMAL']['H20T']['HUMIDITY']))
        self.lineDJIDistance.setText(str(cfg['PREPROCESSING']['THERMAL']['H20T']['DISTANCE']))
        self.lineDJIEmissivity.setText(str(cfg['PREPROCESSING']['THERMAL']['H20T']['EMISSIVITY']))

        # homography
        self.listHomographyMode.setCurrentRow(['UNALIGNED','MANUAL','NGF', 'ECC'].index(cfg['HOMOGRAPHY']['MODE']))
        self.listDOF.setCurrentRow(['affine','perspective'].index(cfg['HOMOGRAPHY']['DOF']))
        self.checkManualHistEq.setChecked(cfg['HOMOGRAPHY']['MANUAL_PARAMS']['HIST_EQ'])
        self.lineManualImagePairIndex.setText(str(cfg['HOMOGRAPHY']['MANUAL_PARAMS']['IMAGE_PAIR_INDEX']))
        self.checkNGFHistEq.setChecked(cfg['HOMOGRAPHY']['NGF_PARAMS']['HIST_EQ'])
        self.checkNGFTwoWayLoss.setChecked(cfg['HOMOGRAPHY']['NGF_PARAMS']['TWO_WAY_LOSS'])
        self.lineNGFBatchSize.setText(str(cfg['HOMOGRAPHY']['NGF_PARAMS']['BATCH_SIZE']))
        self.lineNGFDownscale.setText(str(cfg['HOMOGRAPHY']['NGF_PARAMS']['DOWNSCALE']))
        self.lineNGFLearningRate.setText(str(cfg['HOMOGRAPHY']['NGF_PARAMS']['LEARNING_RATE']))
        self.lineNGFLevels.setText(str(cfg['HOMOGRAPHY']['NGF_PARAMS']['LEVELS']))
        self.lineNGFNumIters.setText(str(cfg['HOMOGRAPHY']['NGF_PARAMS']['N_ITERS'])) 
        # self.lineNGFMILowerBound.setText(str(cfg['HOMOGRAPHY']['NGF_PARAMS']['MI_LOWER_BOUND'])) # deleted
        self.checkNGFSystematic.setChecked(cfg['HOMOGRAPHY']['NGF_PARAMS']['SYSTEMATIC_SAMPLE'])

        # ODM
        with open('resources/ODM_params.yml', 'rb') as f:
            ODM_params = yaml.safe_load(f.read())
        for param in ODM_params.keys():
            find_param = param.replace('-','_')
            cb = self.findChild(QtWidgets.QCheckBox, f"c_{find_param}")
            cb.setChecked(cb.text()!='')
        self.l_matcher_neighbors.setText(str(cfg["ODM"]['matcher-neighbors']))
        self.l_orthophoto_resolution.setText(str(cfg["ODM"]['orthophoto-resolution']))


    
        

    def startProcessingButtonClicked(self):
        '''
        Call function to start 
        '''
        # check if paths are valid
        if not os.path.exists(self.lineProjectDir.text()):
            print("Error! That project path is invalid.")
        if self.lineRGBDir.text() != '' and not os.path.exists(self.lineRGBDir.text()):
            print("Error! The RGB data path is invalid.")
        if self.lineThermalDir.text() != '' and not os.path.exists(self.lineThermalDir.text()):
            print("Error! The Thermal data path is invalid.")

        # pipeline_process('configs/combined.yml')
        # self.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
        # self.p.start("python", ['dummy_script.py'])
        print("Starting processing.")
        # self.pipeline_process.start("python", ['-u', 'mosaic_cfg_trial.py', 'configs/combined.yml'])
        self.pipeline_process.start('./launcher.bat')



    

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    #open qss file

    with open("stylesheets/ConsoleStyle.qss",'r') as f: # ConsoleStyle or Ubuntu
        qss = f.read()
        app.setStyleSheet(qss)
    print('a')
    print('a')
    ui = PipelineTool()
    ui.show()
    sys.exit(app.exec_())
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on April 09, 2025, at 19:16
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'tutorial'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\dash\\repos\\ai-assistance\\tutorial_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "step1" ---
    instruct = visual.TextBox2(
         win, text='Welcome to (insert lab name), I am (insert RA name), are you (participants name). (sentence about where participants will place belongings). \nDo you need corrective lenses for sight, and are they on your person?\nIf yes: please, put them on for this experiment\n\nClick NEXT to start the task!', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0.1), draggable=False,      letterHeight=0.05,
         size=(1.2, 0.7), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='instruct',
         depth=0, autoLog=True,
    )
    instruct_mouse = event.Mouse(win=win)
    x, y = [None, None]
    instruct_mouse.mouseClock = core.Clock()
    next_text = visual.TextBox2(
         win, text='NEXT', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, -0.4), draggable=False,      letterHeight=0.05,
         size=(0.15, 0.15), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='next_text',
         depth=-2, autoLog=True,
    )
    
    # --- Initialize components for Routine "step2" ---
    instruct2 = visual.TextBox2(
         win, text=' “I am going to go over the informed consent sheet (hand them consent form). Please stop me and ask any questions regarding the consent form at any point. Today you going to complete a pre-test questionnaire on introductory statistical concepts, then watch a 12 minute video. You will be asked to fill out a series of questions regarding the contents of the video. It will take approximately 75 minutes to complete this study. There are minimal risk to participation in this research, however you can stop this experiment at anytime; while participation may not specifically benefit you, we believe it will improve general understanding of how humans learn statistical concepts. Your responses will be confidential and completely anonymous. For you participation you will be rewarded SONA credits.”\n', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0.1), draggable=False,      letterHeight=0.05,
         size=(1.5, 0.8), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='instruct2',
         depth=0, autoLog=True,
    )
    instruct_mouse2 = event.Mouse(win=win)
    x, y = [None, None]
    instruct_mouse2.mouseClock = core.Clock()
    next_text2 = visual.TextBox2(
         win, text='NEXT', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, -0.4), draggable=False,      letterHeight=0.05,
         size=(0.15, 0.15), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='next_text2',
         depth=-2, autoLog=True,
    )
    
    # --- Initialize components for Routine "step3_pretest" ---
    win.allowStencil = True
    pretest = visual.Form(win=win, name='pretest',
        items='pretest.xlsx',
        textHeight=0.03,
        font='Open Sans',
        randomize=False,
        style='dark',
        fillColor=None, borderColor=None, itemColor='white', 
        responseColor='white', markerColor='red', colorSpace='rgb', 
        size=(1.75, 0.6),
        pos=(0, 0),
        itemPadding=0.05,
        depth=0
    )
    step3_mouse = event.Mouse(win=win)
    x, y = [None, None]
    step3_mouse.mouseClock = core.Clock()
    step3_next = visual.TextBox2(
         win, text='NEXT', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, -0.4), draggable=False,      letterHeight=0.05,
         size=(0.15, 0.15), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='step3_next',
         depth=-2, autoLog=True,
    )
    
    # --- Initialize components for Routine "step4_qestion1" ---
    win.allowStencil = True
    question1 = visual.Form(win=win, name='question1',
        items='question1.xlsx',
        textHeight=0.03,
        font='Open Sans',
        randomize=False,
        style='dark',
        fillColor=None, borderColor=None, itemColor='white', 
        responseColor='white', markerColor='red', colorSpace='rgb', 
        size=(1.75, 0.8),
        pos=(0, +0.1),
        itemPadding=0.05,
        depth=0
    )
    step4_mouse = event.Mouse(win=win)
    x, y = [None, None]
    step4_mouse.mouseClock = core.Clock()
    step4_next = visual.TextBox2(
         win, text='NEXT', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, -0.4), draggable=False,      letterHeight=0.05,
         size=(0.15, 0.15), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='step4_next',
         depth=-2, autoLog=True,
    )
    ai_dialog = visual.TextBox2(
         win, text='Ask a question of the ai.\n\nQuestion> ', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(+0.6, -0.3), draggable=False,      letterHeight=0.05,
         size=(1.0, 0.3), borderWidth=2.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='top-left',
         anchor='center', overflow='scroll',
         fillColor=[1.0000, 1.0000, 1.0000], borderColor=[-1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='ai_dialog',
         depth=-3, autoLog=True,
    )
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "step1" ---
    # create an object to store info about Routine step1
    step1 = data.Routine(
        name='step1',
        components=[instruct, instruct_mouse, next_text],
    )
    step1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instruct.reset()
    # setup some python lists for storing info about the instruct_mouse
    instruct_mouse.x = []
    instruct_mouse.y = []
    instruct_mouse.leftButton = []
    instruct_mouse.midButton = []
    instruct_mouse.rightButton = []
    instruct_mouse.time = []
    instruct_mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    next_text.reset()
    # store start times for step1
    step1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    step1.tStart = globalClock.getTime(format='float')
    step1.status = STARTED
    thisExp.addData('step1.started', step1.tStart)
    step1.maxDuration = None
    # keep track of which components have finished
    step1Components = step1.components
    for thisComponent in step1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "step1" ---
    step1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruct* updates
        
        # if instruct is starting this frame...
        if instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruct.frameNStart = frameN  # exact frame index
            instruct.tStart = t  # local t and not account for scr refresh
            instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruct, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruct.started')
            # update status
            instruct.status = STARTED
            instruct.setAutoDraw(True)
        
        # if instruct is active this frame...
        if instruct.status == STARTED:
            # update params
            pass
        # *instruct_mouse* updates
        
        # if instruct_mouse is starting this frame...
        if instruct_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruct_mouse.frameNStart = frameN  # exact frame index
            instruct_mouse.tStart = t  # local t and not account for scr refresh
            instruct_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruct_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('instruct_mouse.started', t)
            # update status
            instruct_mouse.status = STARTED
            instruct_mouse.mouseClock.reset()
            prevButtonState = instruct_mouse.getPressed()  # if button is down already this ISN'T a new click
        if instruct_mouse.status == STARTED:  # only update if started and not finished!
            buttons = instruct_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(next_text, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(instruct_mouse):
                            gotValidClick = True
                            instruct_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        instruct_mouse.clicked_name.append(None)
                    x, y = instruct_mouse.getPos()
                    instruct_mouse.x.append(x)
                    instruct_mouse.y.append(y)
                    buttons = instruct_mouse.getPressed()
                    instruct_mouse.leftButton.append(buttons[0])
                    instruct_mouse.midButton.append(buttons[1])
                    instruct_mouse.rightButton.append(buttons[2])
                    instruct_mouse.time.append(instruct_mouse.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *next_text* updates
        
        # if next_text is starting this frame...
        if next_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            next_text.frameNStart = frameN  # exact frame index
            next_text.tStart = t  # local t and not account for scr refresh
            next_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(next_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'next_text.started')
            # update status
            next_text.status = STARTED
            next_text.setAutoDraw(True)
        
        # if next_text is active this frame...
        if next_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            step1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in step1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "step1" ---
    for thisComponent in step1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for step1
    step1.tStop = globalClock.getTime(format='float')
    step1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('step1.stopped', step1.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('instruct_mouse.x', instruct_mouse.x)
    thisExp.addData('instruct_mouse.y', instruct_mouse.y)
    thisExp.addData('instruct_mouse.leftButton', instruct_mouse.leftButton)
    thisExp.addData('instruct_mouse.midButton', instruct_mouse.midButton)
    thisExp.addData('instruct_mouse.rightButton', instruct_mouse.rightButton)
    thisExp.addData('instruct_mouse.time', instruct_mouse.time)
    thisExp.addData('instruct_mouse.clicked_name', instruct_mouse.clicked_name)
    thisExp.nextEntry()
    # the Routine "step1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "step2" ---
    # create an object to store info about Routine step2
    step2 = data.Routine(
        name='step2',
        components=[instruct2, instruct_mouse2, next_text2],
    )
    step2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instruct2.reset()
    # setup some python lists for storing info about the instruct_mouse2
    instruct_mouse2.x = []
    instruct_mouse2.y = []
    instruct_mouse2.leftButton = []
    instruct_mouse2.midButton = []
    instruct_mouse2.rightButton = []
    instruct_mouse2.time = []
    instruct_mouse2.clicked_name = []
    gotValidClick = False  # until a click is received
    next_text2.reset()
    # store start times for step2
    step2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    step2.tStart = globalClock.getTime(format='float')
    step2.status = STARTED
    thisExp.addData('step2.started', step2.tStart)
    step2.maxDuration = None
    # keep track of which components have finished
    step2Components = step2.components
    for thisComponent in step2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "step2" ---
    step2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruct2* updates
        
        # if instruct2 is starting this frame...
        if instruct2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruct2.frameNStart = frameN  # exact frame index
            instruct2.tStart = t  # local t and not account for scr refresh
            instruct2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruct2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruct2.started')
            # update status
            instruct2.status = STARTED
            instruct2.setAutoDraw(True)
        
        # if instruct2 is active this frame...
        if instruct2.status == STARTED:
            # update params
            pass
        # *instruct_mouse2* updates
        
        # if instruct_mouse2 is starting this frame...
        if instruct_mouse2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruct_mouse2.frameNStart = frameN  # exact frame index
            instruct_mouse2.tStart = t  # local t and not account for scr refresh
            instruct_mouse2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruct_mouse2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('instruct_mouse2.started', t)
            # update status
            instruct_mouse2.status = STARTED
            instruct_mouse2.mouseClock.reset()
            prevButtonState = instruct_mouse2.getPressed()  # if button is down already this ISN'T a new click
        if instruct_mouse2.status == STARTED:  # only update if started and not finished!
            buttons = instruct_mouse2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(next_text2, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(instruct_mouse2):
                            gotValidClick = True
                            instruct_mouse2.clicked_name.append(obj.name)
                    if not gotValidClick:
                        instruct_mouse2.clicked_name.append(None)
                    x, y = instruct_mouse2.getPos()
                    instruct_mouse2.x.append(x)
                    instruct_mouse2.y.append(y)
                    buttons = instruct_mouse2.getPressed()
                    instruct_mouse2.leftButton.append(buttons[0])
                    instruct_mouse2.midButton.append(buttons[1])
                    instruct_mouse2.rightButton.append(buttons[2])
                    instruct_mouse2.time.append(instruct_mouse2.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *next_text2* updates
        
        # if next_text2 is starting this frame...
        if next_text2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            next_text2.frameNStart = frameN  # exact frame index
            next_text2.tStart = t  # local t and not account for scr refresh
            next_text2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(next_text2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'next_text2.started')
            # update status
            next_text2.status = STARTED
            next_text2.setAutoDraw(True)
        
        # if next_text2 is active this frame...
        if next_text2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            step2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in step2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "step2" ---
    for thisComponent in step2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for step2
    step2.tStop = globalClock.getTime(format='float')
    step2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('step2.stopped', step2.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('instruct_mouse2.x', instruct_mouse2.x)
    thisExp.addData('instruct_mouse2.y', instruct_mouse2.y)
    thisExp.addData('instruct_mouse2.leftButton', instruct_mouse2.leftButton)
    thisExp.addData('instruct_mouse2.midButton', instruct_mouse2.midButton)
    thisExp.addData('instruct_mouse2.rightButton', instruct_mouse2.rightButton)
    thisExp.addData('instruct_mouse2.time', instruct_mouse2.time)
    thisExp.addData('instruct_mouse2.clicked_name', instruct_mouse2.clicked_name)
    thisExp.nextEntry()
    # the Routine "step2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "step3_pretest" ---
    # create an object to store info about Routine step3_pretest
    step3_pretest = data.Routine(
        name='step3_pretest',
        components=[pretest, step3_mouse, step3_next],
    )
    step3_pretest.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the step3_mouse
    step3_mouse.x = []
    step3_mouse.y = []
    step3_mouse.leftButton = []
    step3_mouse.midButton = []
    step3_mouse.rightButton = []
    step3_mouse.time = []
    step3_mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    step3_next.reset()
    # store start times for step3_pretest
    step3_pretest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    step3_pretest.tStart = globalClock.getTime(format='float')
    step3_pretest.status = STARTED
    thisExp.addData('step3_pretest.started', step3_pretest.tStart)
    step3_pretest.maxDuration = None
    # keep track of which components have finished
    step3_pretestComponents = step3_pretest.components
    for thisComponent in step3_pretest.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "step3_pretest" ---
    step3_pretest.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *pretest* updates
        
        # if pretest is starting this frame...
        if pretest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pretest.frameNStart = frameN  # exact frame index
            pretest.tStart = t  # local t and not account for scr refresh
            pretest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pretest, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'pretest.started')
            # update status
            pretest.status = STARTED
            pretest.setAutoDraw(True)
        
        # if pretest is active this frame...
        if pretest.status == STARTED:
            # update params
            pass
        # *step3_mouse* updates
        
        # if step3_mouse is starting this frame...
        if step3_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            step3_mouse.frameNStart = frameN  # exact frame index
            step3_mouse.tStart = t  # local t and not account for scr refresh
            step3_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(step3_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('step3_mouse.started', t)
            # update status
            step3_mouse.status = STARTED
            step3_mouse.mouseClock.reset()
            prevButtonState = step3_mouse.getPressed()  # if button is down already this ISN'T a new click
        if step3_mouse.status == STARTED:  # only update if started and not finished!
            buttons = step3_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(step3_next, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(step3_mouse):
                            gotValidClick = True
                            step3_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        step3_mouse.clicked_name.append(None)
                    x, y = step3_mouse.getPos()
                    step3_mouse.x.append(x)
                    step3_mouse.y.append(y)
                    buttons = step3_mouse.getPressed()
                    step3_mouse.leftButton.append(buttons[0])
                    step3_mouse.midButton.append(buttons[1])
                    step3_mouse.rightButton.append(buttons[2])
                    step3_mouse.time.append(step3_mouse.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *step3_next* updates
        
        # if step3_next is starting this frame...
        if step3_next.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            step3_next.frameNStart = frameN  # exact frame index
            step3_next.tStart = t  # local t and not account for scr refresh
            step3_next.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(step3_next, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'step3_next.started')
            # update status
            step3_next.status = STARTED
            step3_next.setAutoDraw(True)
        
        # if step3_next is active this frame...
        if step3_next.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            step3_pretest.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in step3_pretest.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "step3_pretest" ---
    for thisComponent in step3_pretest.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for step3_pretest
    step3_pretest.tStop = globalClock.getTime(format='float')
    step3_pretest.tStopRefresh = tThisFlipGlobal
    thisExp.addData('step3_pretest.stopped', step3_pretest.tStop)
    pretest.addDataToExp(thisExp, 'rows')
    pretest.autodraw = False
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('step3_mouse.x', step3_mouse.x)
    thisExp.addData('step3_mouse.y', step3_mouse.y)
    thisExp.addData('step3_mouse.leftButton', step3_mouse.leftButton)
    thisExp.addData('step3_mouse.midButton', step3_mouse.midButton)
    thisExp.addData('step3_mouse.rightButton', step3_mouse.rightButton)
    thisExp.addData('step3_mouse.time', step3_mouse.time)
    thisExp.addData('step3_mouse.clicked_name', step3_mouse.clicked_name)
    thisExp.nextEntry()
    # the Routine "step3_pretest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "step4_qestion1" ---
    # create an object to store info about Routine step4_qestion1
    step4_qestion1 = data.Routine(
        name='step4_qestion1',
        components=[question1, step4_mouse, step4_next, ai_dialog],
    )
    step4_qestion1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the step4_mouse
    step4_mouse.x = []
    step4_mouse.y = []
    step4_mouse.leftButton = []
    step4_mouse.midButton = []
    step4_mouse.rightButton = []
    step4_mouse.time = []
    step4_mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    step4_next.reset()
    ai_dialog.reset()
    # store start times for step4_qestion1
    step4_qestion1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    step4_qestion1.tStart = globalClock.getTime(format='float')
    step4_qestion1.status = STARTED
    thisExp.addData('step4_qestion1.started', step4_qestion1.tStart)
    step4_qestion1.maxDuration = None
    # keep track of which components have finished
    step4_qestion1Components = step4_qestion1.components
    for thisComponent in step4_qestion1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "step4_qestion1" ---
    step4_qestion1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *question1* updates
        
        # if question1 is starting this frame...
        if question1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            question1.frameNStart = frameN  # exact frame index
            question1.tStart = t  # local t and not account for scr refresh
            question1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(question1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'question1.started')
            # update status
            question1.status = STARTED
            question1.setAutoDraw(True)
        
        # if question1 is active this frame...
        if question1.status == STARTED:
            # update params
            pass
        # *step4_mouse* updates
        
        # if step4_mouse is starting this frame...
        if step4_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            step4_mouse.frameNStart = frameN  # exact frame index
            step4_mouse.tStart = t  # local t and not account for scr refresh
            step4_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(step4_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('step4_mouse.started', t)
            # update status
            step4_mouse.status = STARTED
            step4_mouse.mouseClock.reset()
            prevButtonState = step4_mouse.getPressed()  # if button is down already this ISN'T a new click
        if step4_mouse.status == STARTED:  # only update if started and not finished!
            buttons = step4_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(step3_next, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(step4_mouse):
                            gotValidClick = True
                            step4_mouse.clicked_name.append(obj.name)
                    if not gotValidClick:
                        step4_mouse.clicked_name.append(None)
                    x, y = step4_mouse.getPos()
                    step4_mouse.x.append(x)
                    step4_mouse.y.append(y)
                    buttons = step4_mouse.getPressed()
                    step4_mouse.leftButton.append(buttons[0])
                    step4_mouse.midButton.append(buttons[1])
                    step4_mouse.rightButton.append(buttons[2])
                    step4_mouse.time.append(step4_mouse.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # *step4_next* updates
        
        # if step4_next is starting this frame...
        if step4_next.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            step4_next.frameNStart = frameN  # exact frame index
            step4_next.tStart = t  # local t and not account for scr refresh
            step4_next.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(step4_next, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'step4_next.started')
            # update status
            step4_next.status = STARTED
            step4_next.setAutoDraw(True)
        
        # if step4_next is active this frame...
        if step4_next.status == STARTED:
            # update params
            pass
        
        # *ai_dialog* updates
        
        # if ai_dialog is starting this frame...
        if ai_dialog.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            ai_dialog.frameNStart = frameN  # exact frame index
            ai_dialog.tStart = t  # local t and not account for scr refresh
            ai_dialog.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ai_dialog, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ai_dialog.started')
            # update status
            ai_dialog.status = STARTED
            ai_dialog.setAutoDraw(True)
        
        # if ai_dialog is active this frame...
        if ai_dialog.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            step4_qestion1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in step4_qestion1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "step4_qestion1" ---
    for thisComponent in step4_qestion1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for step4_qestion1
    step4_qestion1.tStop = globalClock.getTime(format='float')
    step4_qestion1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('step4_qestion1.stopped', step4_qestion1.tStop)
    question1.addDataToExp(thisExp, 'rows')
    question1.autodraw = False
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('step4_mouse.x', step4_mouse.x)
    thisExp.addData('step4_mouse.y', step4_mouse.y)
    thisExp.addData('step4_mouse.leftButton', step4_mouse.leftButton)
    thisExp.addData('step4_mouse.midButton', step4_mouse.midButton)
    thisExp.addData('step4_mouse.rightButton', step4_mouse.rightButton)
    thisExp.addData('step4_mouse.time', step4_mouse.time)
    thisExp.addData('step4_mouse.clicked_name', step4_mouse.clicked_name)
    thisExp.addData('ai_dialog.text',ai_dialog.text)
    thisExp.nextEntry()
    # the Routine "step4_qestion1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)

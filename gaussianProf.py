#!/usr/bin/python3
import sys
import glob
import json
import getopt
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


class Block:
    def __init__ (self, block_id, name, start, end):
        self._start = start
        self._end = end
        self._block_id = block_id
        self._kernel_name = name

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._start = start

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, end):
        self._end = end

    @property
    def duration(self):
        return self._end - self._start

    def draw(self, start_offset, stream, height, ax, color, hatch, scaling=1e-6):
        y_lower = stream
        start = (self.start - start_offset) * scaling
        duration = self.duration * scaling
        
        ax.broken_barh([(start, duration)], (y_lower, height), facecolor=color, alpha=0.4, hatch=hatch,edgecolor='k',linewidth=1.0)
        ax.text(start + duration/2,y_lower+height/2,"{:s}:{:d}".format(self._kernel_name, self._block_id),horizontalalignment='center',verticalalignment='center', rotation=90)


class Kernel:
    def __init__ (self, name, block_times, times_per_block=2, n_blocks=2):
        self._n_blocks = 0
        self._blocks = []
        self._name = name
        self.initBlocks(block_times, times_per_block, n_blocks)

    def initBlocks(self, block_times, times_per_block=2, n_blocks=2):
        for block_id in range(n_blocks):
            start_index = times_per_block*block_id
            end_index = start_index + 1
            # Skip blocks with no times
            if block_times[start_index] == 0.0 and block_times[end_index] == 0.0:
                continue
            block = Block(block_id, self._name, block_times[start_index], block_times[end_index]) 
            self._blocks.append(block)
            self._n_blocks = self._n_blocks +1

    @property
    def duration(self):
        return self.end - self.start

    @property
    def start(self):
        if self._n_blocks > 1:
            minStart = sys.float_info.max
            for block in self._blocks:
                if block.start < minStart:
                    minStart = block.start
            return minStart
        else:
            return self._blocks[0].start

    @property
    def end(self):
        if self._n_blocks > 1:
            maxEnd = 0.0
            for block in self._blocks:
                if block.end > maxEnd:
                    maxEnd = block.end
            return maxEnd
        else:
            return self._blocks[0].end

    def draw(self, start_offset, stream, kernel_height, ax, color, hatch, time_scaling):
        for block_id, block in enumerate(self._blocks):
            block_height = kernel_height/self._n_blocks
            block.draw(start_offset, stream + block_id * block_height, block_height, ax, color, hatch, time_scaling)


class Scale:
    def __init__ (self, scale_data):
        self._host_start = scale_data['host_start']
        self._host_end = scale_data['host_end']
        self._ctx_id = scale_data['ctx_id']
        self._kernels = []
        for kernel in scale_data['kernel_names']:
            self._kernels.append(Kernel(kernel, scale_data[kernel], scale_data['times_per_block'], scale_data['n_blocks']))

    @property
    def ctx_id(self):
        return self._ctx_id

    @property
    def durationGPU(self):
        return self.endGPU - self.startGPU

    @property
    def startGPU(self):
        minStart = sys.float_info.max
        for kernel in self._kernels:
            if kernel.start < minStart:
                minStart = kernel.start
        return minStart

    @property
    def endGPU(self):
        maxEnd = 0.0
        for kernel in self._kernels:
            if kernel.end > maxEnd:
                maxEnd = kernel.end
        return maxEnd

    @property
    def durationCPU(self):
        return self._host_end - self._host_start

    @property
    def startCPU(self):
        return self._host_start

    @property
    def endCPU(self):
        return self._host_end

    def draw(self, start_offset, stream, kernel_height, ax, colors, hatches, time_scaling):
        # CPU start
        ax.arrow((self._host_start - start_offset) * time_scaling, stream, 0, 0.7*kernel_height, width=0.02, color='r',alpha=0.4)
        ax.arrow((self._host_start - start_offset) * time_scaling, stream, 0, 0.7*kernel_height, color='k')
        # CPU end
        ax.arrow((self._host_end - start_offset) * time_scaling, stream + 0.7*kernel_height, 0, -0.6*kernel_height, width=0.02, color='b', alpha=0.4)
        ax.arrow((self._host_end - start_offset) * time_scaling, stream + 0.7*kernel_height, 0, -0.6*kernel_height, color='k')
        for kernel_id, kernel in enumerate(self._kernels):
            kernel.draw(start_offset, stream, kernel_height, ax, colors[kernel_id], hatches[kernel_id], time_scaling)


class Frame:
    def __init__ (self):
        # A dict with thread_id as key to aggregate the scales processed by the
        # same CPU thread
        self._scales = {}
        self._n_cpu_threads = 0

    def addScale(self, thread_id, scale_data):
        if not thread_id in self._scales:
            self._scales[thread_id] = []
            self._n_cpu_threads = self._n_cpu_threads + 1
        scale = Scale(scale_data)
        self._scales[thread_id].append(scale)
    
    def sortScales(self):
        # Sort all scales for each thread to ensure that they are in correct
        # order for the further processing
        for thread_id in self._scales.keys():
            self._scales[thread_id].sort(key=lambda scale:scale.startGPU)

    def getParallelScaleDuration(self, target='gpu', scale_id=0):
        minStart = sys.float_info.max
        maxEnd = 0.0
        for thread_id in range(self._n_cpu_threads):
            if target=='gpu':
                if self._scales[thread_id][scale_id].startGPU < minStart:
                    minStart = self._scales[thread_id][scale_id].startGPU
                if self._scales[thread_id][scale_id].endGPU > maxEnd:
                    maxEnd = self._scales[thread_id][scale_id].endGPU
            else:
                if self._scales[thread_id][scale_id].startCPU < minStart:
                    minStart = self._scales[thread_id][scale_id].startCPU
                if self._scales[thread_id][scale_id].endCPU > maxEnd:
                    maxEnd = self._scales[thread_id][scale_id].endCPU
        return maxEnd - minStart

    def getFrameDurations(self, target='gpu', parallelScales=False):
        frameDurations = []
        if parallelScales:
            for scale_id in range(len(self._scales[0])):
                frameDurations.append(self.getParallelScaleDuration(target, scale_id))
        else:
            for thread_id in range(self._n_cpu_threads):
                for scale in self._scales[thread_id]:
                    if target == 'gpu':
                        frameDurations.append(scale.durationGPU)
                    else:
                        frameDurations.append(scale.durationCPU)

        return frameDurations

    def printFrame(self):
        for thread_id in range(self._n_cpu_threads):
            print("Thread {:d}: ".format(thread_id), end = '')
            for scale in self._scales[thread_id]:
                print(" {:3d}".format(scale.ctx_id), end = '')
            print("")

    def getScaleBorders(self, scale_ids):
        minStart = sys.float_info.max
        maxEnd = 0
        for thread_id in range(self._n_cpu_threads):
            for scale_id in scale_ids:
                if self._scales[thread_id][scale_id].startCPU < minStart:
                    minStart = self._scales[thread_id][scale_id].startCPU
                if self._scales[thread_id][scale_id].endCPU > maxEnd:
                    maxEnd = self._scales[thread_id][scale_id].endCPU
        return minStart, maxEnd

    def draw(self, frame_id, scale_ids=[]):
        if not scale_ids:
            scale_ids = [count for count, scale in enumerate(self._scales[0])]

        fig = plt.figure(figsize=[7,2.5])
        title = "GaussianCorrelation: Frame {:d} Scales: [ ".format(frame_id)
        for scale_id in scale_ids:
            title = title + "{:d} ".format(scale_id)
        title = title + "]"

        fig.suptitle(title)
        colors = cm.get_cmap('viridis', 8).colors
        hatches = ['////', '\\\\\\\\', '//', '\\\\', 'x', 'xx','o','O', '.']
        ax = fig.add_subplot(1,1,1)
        labels = []
        
        start_offset, scale_end = self.getScaleBorders(scale_ids)
        # Plotting
        for thread_id in range(self._n_cpu_threads):
            labels.append("Thread "+str(thread_id))
            for scale_id in scale_ids:
                self._scales[thread_id][scale_id].draw(start_offset, thread_id, 1.0, ax, colors, hatches, 1e-6)

        ax.set_ylabel("Streams/CPU_thread")
        ax.set_yticks(np.arange(0.5, self._n_cpu_threads+0.5, 1.0))
        ax.set_ylim((0, self._n_cpu_threads))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Time [ms]")
        ax.set_xlim((0, (scale_end-start_offset)*1e-6))
        ax.grid(True)
        return fig


class Scenario:
    def __init__ (self):
        self._frames = []
        self._n_frames = 0

    def getScaleDurationStats(self, target='gpu', parallelScales=False):
        durations = []
        for frame in self._frames:
            durations.extend(frame.getFrameDurations(target, parallelScales))
        minDur = min(durations)
        maxDur = max(durations)
        avgDur = np.mean(durations)
        return (minDur, maxDur, avgDur)

    def printStats(self):
        gpupar = self.getScaleDurationStats(target='gpu', parallelScales=True)
        gpu = self.getScaleDurationStats(target='gpu', parallelScales=False)
        cpupar = self.getScaleDurationStats(target='cpu', parallelScales=True)
        cpu = self.getScaleDurationStats(target='cpu', parallelScales=False)
        print("{:6s} {:17s} {:10s} {:10s} {:10s}".format("Target", "aggregated scales", "min [ms]", "max [ms]", "avg [ms]"))
        print("{:6s} {:17s} {:<10f} {:<10f} {:<10f}".format("GPU", "yes", gpupar[0] * 1e-6, gpupar[1] * 1e-6, gpupar[2] * 1e-6))
        print("{:6s} {:17s} {:<10f} {:<10f} {:<10f}".format("GPU", "no", gpu[0] * 1e-6, gpu[1] * 1e-6, gpu[2] * 1e-6))
        print("{:6s} {:17s} {:<10f} {:<10f} {:<10f}".format("CPU", "yes", cpupar[0] * 1e-6, cpupar[1] * 1e-6, cpupar[2] * 1e-6))
        print("{:6s} {:17s} {:<10f} {:<10f} {:<10f}".format("CPU", "no", cpu[0] * 1e-6, cpu[1] * 1e-6, cpu[2] * 1e-6))

    def drawFrame(self, frame_id=0, scale_ids=[]):
        if frame_id >= len(self._frames):
            sys.exit("frame_id is out of range")
        return self._frames[frame_id].draw(frame_id, scale_ids)

    def loadData(self, fileNames = []):
        ctxData = []
        for fileName in fileNames:
            with open(fileName) as f1:
                ctxData.append(json.load(f1))

        for ctx in ctxData:
            self.parseData(ctx)
        # Sort the added scales for each frame since the files are not parsed
        # in the correct order 
        for frame in self._frames:
            frame.sortScales()

    def parseData(self, ctx):
        if not self._frames:
            # Skip first frame since we have some allocations and warm up iterations here
            self._frames = [Frame() for frame in ctx['frame'][1:]]
            self._n_frames = len(self._frames)
        if not len(self._frames) == len(ctx['frame'][1:]):
            sys.exit("All threadCtx must have the same number of frames")

        kernel_names = ctx['kernel_names']

        times_per_block = int(ctx['times_per_block'])
        if times_per_block != 2:
            sys.exit("We need exactly two timestamps per block. Start and End Time")

        n_blocks = int(ctx['nof_blocks'])
        if n_blocks < 1:
            sys.exit("We must have at least one block in a kernel")

        thread_id = 0
        if 'thread_id' in ctx:
            thread_id = int(ctx['thread_id'])

        ctx_id = int(ctx['ctx_id'])
        for frame_id, frame in enumerate(ctx['frame'][1:]):
            scale_data = {}
            scale_data['times_per_block'] = times_per_block
            scale_data['n_blocks'] = n_blocks
            scale_data['host_start'] = float(frame['host_start'])
            scale_data['host_end'] = float(frame['host_end'])
            scale_data['kernel_names'] = kernel_names
            scale_data['ctx_id'] = ctx_id
            for kernel in kernel_names:
                scale_data[kernel] = [float(i) for i in frame[kernel]]
            self._frames[frame_id].addScale(thread_id, scale_data)

    def printFrames(self):
        for frame_id, frame in enumerate(self._frames):
            print("Frame id: {:d}".format(frame_id))
            frame.printFrame()
            print("----------------------------------------------------------------------")


if __name__ == "__main__":
    # Show plot of selected frame
    base_directory = "./logs"
    debug = False
    draw = False
    frame_id = 0
    scale_ids = []
    print_stats = False

    options, remainder = getopt.getopt(sys.argv[1:], 'i:pf:s:dh', ['input=', 'debug', 'plot', 'frame=', 'scales=', 'durations', 'help'])

    for opt, arg in options:
        if opt in ('-i', '--input'):
            base_directory = arg
        elif opt == '--debug':
            debug = True
        elif opt in ('-p', '--plot'):
            draw = True
        elif opt in ('-f', '--frame'):
            frame_id = int(arg)
        elif opt in ('-s', '--scales'):
            scale_ids = [int(scale) for scale in arg.split(',')]
        elif opt in ('-p', '--plot'):
            draw = True
        elif opt in ('-d', '--durations'):
            print_stats = True
        else:
            print("Help output of {:s}".format( sys.argv[0]))
            print("{:27s}: {:s}".format("Option", "Describtion"))
            print("{:27s}: {:s}".format("-i <path>/--input=<path>", "Dir of input files"))
            print("{:27s}: {:s}".format("--debug", "Prints the loaded data structure"))
            print("{:27s}: {:s}".format("-p /--plot", "Plots graph, default is all scales of the first frame"))
            print("{:27s}: {:s}".format("-f <nr>/--frame=<nr>", "Frame number to plot"))
            print("{:27s}: {:s}".format("-s <id0,id1...>/--scales", "Scales to plot (comma separated list)"))
            print("{:27s}: {:s}".format("-d/--durations", "Prints statistic of GPU and CPU processing times"))
            print("{:27s}: {:s}".format("-h/--help", "Prints this help outptu"))
            sys.exit("{:s} has finished".format( sys.argv[0]))
            

    filenames = glob.glob(base_directory + "/*.log")
    scen = Scenario()
    scen.loadData(filenames)
    if print_stats:
        scen.printStats()
    if debug:
        scen.printFrames()
    if draw:
        scen.drawFrame(frame_id, scale_ids)
        plt.show()

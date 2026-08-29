import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import ScreenCaptureKit

final class SystemAudioOutput: NSObject, SCStreamOutput, SCStreamDelegate {
    private let output = FileHandle.standardOutput
    private let sampleQueue = DispatchQueue(label: "meeting-translator.system-audio")
    private var stream: SCStream?

    func start() async throws {
        if !CGPreflightScreenCaptureAccess() && !CGRequestScreenCaptureAccess() {
            throw NSError(
                domain: "MeetingTranslator",
                code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "画面収録とシステムオーディオの権限が許可されていません。"
                ]
            )
        }
        let content = try await SCShareableContent.excludingDesktopWindows(
            false,
            onScreenWindowsOnly: false
        )
        guard let display = content.displays.first else {
            throw NSError(
                domain: "MeetingTranslator",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "キャプチャ可能なディスプレイがありません。"]
            )
        }

        let filter = SCContentFilter(
            display: display,
            excludingApplications: [],
            exceptingWindows: []
        )
        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.excludesCurrentProcessAudio = true
        configuration.sampleRate = 48_000
        configuration.channelCount = 2
        configuration.width = 2
        configuration.height = 2
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        configuration.queueDepth = 3

        let captureStream = SCStream(
            filter: filter,
            configuration: configuration,
            delegate: self
        )
        try captureStream.addStreamOutput(
            self,
            type: .audio,
            sampleHandlerQueue: sampleQueue
        )
        stream = captureStream
        try await captureStream.startCapture()
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio, sampleBuffer.isValid else { return }
        guard let formatDescription = sampleBuffer.formatDescription else { return }
        let format = AVAudioFormat(cmAudioFormatDescription: formatDescription)
        let frameCount = AVAudioFrameCount(sampleBuffer.numSamples)
        guard frameCount > 0 else { return }
        guard let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        ) else { return }
        pcmBuffer.frameLength = frameCount

        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer,
            at: 0,
            frameCount: Int32(frameCount),
            into: pcmBuffer.mutableAudioBufferList
        )
        guard status == noErr, let channels = pcmBuffer.floatChannelData else { return }

        let channelCount = max(1, Int(format.channelCount))
        var mono = [Float](repeating: 0, count: Int(frameCount))
        for channel in 0..<channelCount {
            let source = channels[channel]
            for frame in 0..<Int(frameCount) {
                mono[frame] += source[frame] / Float(channelCount)
            }
        }
        mono.withUnsafeBytes { bytes in
            output.write(Data(bytes))
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write(
            Data("ScreenCaptureKit: \(error.localizedDescription)\n".utf8)
        )
        exit(2)
    }
}

@main
struct SystemAudioCaptureMain {
    static func main() async {
        let capture = SystemAudioOutput()
        do {
            try await capture.start()
            await withCheckedContinuation { (_: CheckedContinuation<Void, Never>) in }
        } catch {
            FileHandle.standardError.write(
                Data("ScreenCaptureKit: \(error.localizedDescription)\n".utf8)
            )
            exit(1)
        }
    }
}

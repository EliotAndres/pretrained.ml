var app = angular.module('app', ['tableSort', 'pathgather.popeye', 'ngFileUpload']);

var mainUrl = 'http://149.202.173.21';

app
    .controller('MainController', function MainController($scope, $http, Popeye, Upload, $timeout) {
        var socket = io.connect(mainUrl + ':8091');
        $scope.serverUrl = mainUrl + ':8091/outputs/';
        $scope.models = [];
        $scope.form = {};
        $scope.icons = {'Computer Vision': 'computer_vision.png',
            'Natural Language Processing': 'nlp.png'};

        var modals = {'Object Recognition': 'imageModal.html',
            'Semantic Segmentation': 'imageModal.html',
            'Sentiment Analysis': 'nlpModal.html',
            'Object Detection': 'imageModal.html'};

        var trackMixpanel = function (name, data) {
            if (location.hostname !== "localhost") {
                mixpanel.track(name, data);
            }
        };
        trackMixpanel("Page Opened");

        $http({
            method: 'GET',
            url: 'models.yaml'
        }).then(function successCallback(response) {
            $scope.models = jsyaml.load(response.data).models;
            // For dev purposes
            //$scope.openModal($scope.models[0]);
        }, function errorCallback(error) {
            console.error(error);
        });

        socket.on('connect', function() {
            console.log('Connected, sessionID is ' + socket.id);
            $scope.sessionId = socket.id;
        });

        socket.on('finished_job', function(data) {
            // If the user closed the modal and opened a new one before task returned
            if (data.taskId === $scope.currentTaskId) {
                $timeout(function () {
                    $scope.loading = false;
                    $scope.predictions = JSON.parse(data.predictions);
                });
            }
        });

        socket.on('queue_status', function(data) {
            var taskIds = JSON.parse(data.data);
            $timeout(function () {
                $scope.queuePosition = taskIds.indexOf($scope.currentTaskId)
            });
        });

        $scope.openModal = function (model) {
            trackMixpanel("Opened Modal", {"model": model.name});

            $scope.currentModel = model;
            var modal = Popeye.openModal({
                templateUrl: 'views/' + modals[model.subtype],
                scope: $scope,
                modalClass: 'demo-modal'
            });

            modal.closed.then(function() {
                $scope.reset();
            });
        };

        $scope.reset = function () {
            // TODO: move these variables to a modalStatus object
            $scope.image = undefined;
            $scope.text = undefined;
            $scope.predictions = undefined;
            $scope.uploadProgress = undefined;
            $scope.currentTaskId = undefined;
            $scope.queuePosition = undefined;
        };

        $scope.uploadImage = function (files, file) {
            Upload.upload({
                url: mainUrl + ':8091/' + $scope.currentModel.demoUrl,
                data: {file: file, 'sessionId': $scope.sessionId}
            }).progress(function(event) {
                $scope.uploadProgress = parseInt(100.0 * event.loaded / event.total);
            }).then(function(response) {
                $scope.currentTaskId = response.data.taskId
                // TODO notify that request has been added to queue
            });
        };

        $scope.sendText = function (text) {
            $scope.loading = true;
            $scope.predictions = undefined;

            $http({
                method: 'POST',
                url: mainUrl + ':8091/' + $scope.currentModel.demoUrl,
                data: 'text=' + text + '&sessionId=' + $scope.sessionId,
                headers: {'Content-Type': 'application/x-www-form-urlencoded'}
            }).then(function (response) {
                // TODO notify that request has been added to queue
                $scope.currentTaskId = response.data.taskId

            }).catch(function (error) {
                $scope.loading = false;
                console.log(error)
            });
        }
    }).filter('abs', function() {
    return function(num) {
        return Math.abs(num);
    }
});

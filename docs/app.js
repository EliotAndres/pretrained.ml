var app = angular.module('app', ['tableSort', 'pathgather.popeye', 'ngFileUpload']);

app
    .controller('MainController', function MainController($scope, $http, Popeye, Upload, $timeout) {
        var socket = io.connect('http://s2.ndres.me:8091');
        $scope.serverUrl = 'http://s2.ndres.me:8091/outputs/';
        $scope.models = [];
        $scope.form = {};
        $scope.icons = {'Computer Vision': 'computer_vision.png',
            'Natural Language Processing': 'nlp.png'};

        var modals = {'Object Recognition': 'imageModal.html',
            'Semantic Segmentation': 'imageModal.html',
            'Sentiment Analysis': 'nlpModal.html'};

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
            console.log('sessionID ' + socket.id);
            $scope.sessionId = socket.id;
        });

        socket.on('finished_job', function(data) {
            console.log(JSON.parse(data.data));
            $scope.loading = false;

            $timeout(function () {
                $scope.predictions = JSON.parse(data.data);
                console.log($scope.predictions[0])
            });
        });

        $scope.openModal = function (model) {
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
            $scope.image = undefined;
            $scope.text = undefined;
            $scope.predictions = undefined;
            $scope.uploadProgress = undefined;
        };

        $scope.uploadImage = function (files, file) {
            Upload.upload({
                url: 'http://s2.ndres.me:8091/' + $scope.currentModel.demoUrl,
                data: {file: file, 'sessionId': $scope.sessionId}
            }).progress(function(event) {
                $scope.uploadProgress = parseInt(100.0 * event.loaded / event.total);
            }).then(function(data, status, headers, config) {
                // TODO notify that request has been added to queue
            });
        };

        $scope.sendText = function (text) {
            $scope.loading = true;
            $scope.predictions = undefined;

            $http({
                method: 'POST',
                url: 'http://s2.ndres.me:8091/' + $scope.currentModel.demoUrl,
                data: 'text=' + text,
                headers: {'Content-Type': 'application/x-www-form-urlencoded'}
            }).then(function (response) {
                // TODO notify that request has been added to queue
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

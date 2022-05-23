function mkdir_for_file( file )
%MKDIR_FOR_FILE  Creates necessary directory structure for a given file.
%
%   mkdir_for_file( full_filename )
%
%   Creates (sub)directory structure necessary for creating the file. It does
%   not complain if some or all subdirectories allready exist.

% (c) 2007-11-27 Martin Matousek, Czech Technical University in Prague
% Last change: $Date: 2017-06-12 17:31:54 +0200$
%              $Revision: c720398$

path = fileparts( file );

if( isempty( path ) ), return; end

wstate = warning( 'off', 'MATLAB:MKDIR:DirectoryExists' );

[success, message, msgid] = mkdir( path );

warning( wstate );

if( ~success ), error( msgid, message ); end

if( ~exist( path, 'dir' ) )
  error( 'mkdir_for_file:failed', ...
         'Make directory %s failed. (%s)', path, message );
end
